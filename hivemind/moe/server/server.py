from __future__ import annotations

import multiprocessing as mp
import random
import threading
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn

from hivemind.dht import DHT
from hivemind.moe.expert_uid import UID_DELIMITER

from hivemind.moe.server.checkpoints import CheckpointSaver, is_directory, load_expert, load_expert_from_hf
from hivemind.moe.server.module_wrapper import ModuleWrapper

from hivemind.moe.server.bnb_quantization import quantization, init_empty_weights

from hivemind.moe.server.connection_handler import ConnectionHandler
from hivemind.moe.server.dht_handler import DHTHandlerThread, get_experts
from hivemind.moe.server.layers import (
    add_custom_models_from_file,
    name_to_block,
    name_to_input,
)
from hivemind.moe.server.layers.optim import ClippingWrapper
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.moe.server.runtime import Runtime
from hivemind.p2p import PeerInfo
from hivemind.proto.runtime_pb2 import CompressionType
from hivemind.utils.logging import get_logger
from hivemind.utils.tensor_descr import DUMMY_BATCH_SIZE, BatchTensorDescriptor

logger = get_logger(__name__)


class Server(threading.Thread):
    """
    Server allows you to host "experts" - pytorch subnetworks that can be accessed remotely by peers.
    After creation, a server should be started: see Server.run or Server.run_in_background.

    A working server does two things:
     - processes incoming forward/backward requests via Runtime (created by the server)
     - publishes updates to expert status every :update_period: seconds

    :type dht: an instance of hivemind.DHT. Server will use DHT for all network interactions.
    :param module_backends: dict{expert uid (str) : ModuleBackend} for all expert hosted by this server.
    :param num_connection_handlers: maximum number of simultaneous requests. Please note that the default value of 1
        if too small for normal functioning, we recommend 4 handlers per expert backend.
    :param update_period: how often will server attempt to publish its state (i.e. experts) to the DHT;
        if dht is None, this parameter is ignored.
    :param expiration: when server declares its experts to the DHT, these entries will expire after this many seconds
    :param start: if True, the server will immediately start as a background thread and returns control after server
        is ready (see .ready below)
    """

    def __init__(
        self,
        dht: DHT,
        module_backends: Dict[str, ModuleBackend],
        num_connection_handlers: int = 1,
        update_period: float = 30,
        expiration: Optional[float] = None,
        start=False,
        checkpoint_dir: str =None,
        **kwargs,
    ):
        super().__init__()
        self.dht, self.module_backends, self.update_period = dht, module_backends, update_period

        self.conn_handlers = [ConnectionHandler(dht, self.module_backends) for _ in range(num_connection_handlers)]
        if checkpoint_dir is not None:
            self.checkpoint_saver = CheckpointSaver(module_backends, checkpoint_dir, update_period)
        else:
            self.checkpoint_saver = None
        
        self.runtime = Runtime(self.module_backends, **kwargs)

        if self.module_backends:
            self.dht_handler_thread = DHTHandlerThread(
                module_backends=self.module_backends,
                dht=self.dht,
                update_period=self.update_period,
                expiration=expiration,
                daemon=True,
            )

        if start:
            self.run_in_background(await_ready=True)

    @classmethod
    def create(
        cls,
        num_experts: int = 1,
        num_layers: int = 1,
        layers_index_start: int = 0,
        hugginface_rep: str = None,
        checkpoint_dir: Optional[Path] = None,
        load_in_4bit: bool = False,
        expert_pattern: str = None,
        expert_cls="ffn",
        hidden_dim=1024,
        optim_cls=None,
        scheduler: str = "none",
        clip_grad_norm=None,
        num_handlers=None,
        min_batch_size=1,
        max_batch_size=2048,
        device=None,
        initial_peers=(),
        compression=CompressionType.BLOCKWISE_8BIT,
        stats_report_interval: Optional[int] = None,
        custom_module_path=None,
        update_period: float = 30,
        expiration: Optional[float] = None,
        *,
        start: bool,
        **kwargs,
    ) -> Server:
        """
        Instantiate a server with several identical modules. See argparse comments below for details

        :param num_experts: num experts per layer
        :param num_layers: total layers
        :param hugginface_rep: huggingface rep to load weights
        :param load_in_4bit: enable bnb 4bit
        :param expert_pattern: a string pattern of expert uids,  example: myprefix.\{layer_id\}.\{expert_id\}
        :param expert_cls: expert type from hivemind.moe.server.layers, e.g. 'ffn' or 'transformer';
        :param hidden_dim: main dimension for expert_cls
        :param num_handlers: server will use this many parallel processes to handle incoming requests
        :param min_batch_size: total num examples in the same batch will be greater than this value
        :param max_batch_size: total num examples in the same batch will not exceed this value
        :param device: all experts will use this device in torch notation; default: cuda if available else cpu

        :param optim_cls: uses this optimizer to train all experts
        :param scheduler: if not `none`, the name of the expert LR scheduler
        :param num_warmup_steps: the number of warmup steps for LR schedule
        :param num_training_steps: the total number of steps for LR schedule
        :param clip_grad_norm: maximum gradient norm used for clipping
        :param initial_peers: multiaddrs of one or more active DHT peers (if you want to join an existing DHT)


        :param compression: if specified, use this compression to pack all inputs, outputs and gradients by all experts
            hosted on this server. For a more fine-grained compression, start server in python and specify compression
            for each BatchTensorProto in ModuleBackend for the respective experts.

        :param start: if True, starts server right away and returns when server is ready for requests
        :param stats_report_interval: interval between two reports of batch processing performance statistics
        :param kwargs: any other params will be forwarded to DHT upon creation
        """
        if custom_module_path is not None:
            add_custom_models_from_file(custom_module_path)
        assert expert_cls in name_to_block

        dht = DHT(initial_peers=initial_peers, start=True, **kwargs)
        visible_maddrs_str = [str(a) for a in dht.get_visible_maddrs()]
        logger.info(f"Running DHT node on {visible_maddrs_str}, initial peers = {initial_peers}")

        assert not (expert_pattern is None), "Please provide expert_pattern"

        #if checkpoint_dir is not None:
            #expert_uids = 
        #else:
        expert_uids: List[ExpertUID] =_generate_uids(num_experts, num_layers, layers_index_start, expert_pattern, dht)
        uids_to_generate = num_experts * num_layers
        logger.info(f"Generating {uids_to_generate} experts from pattern {expert_pattern}")

        num_experts = len(expert_uids)
        num_handlers = num_handlers if num_handlers is not None else num_experts * 8
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        sample_input = name_to_input[expert_cls](DUMMY_BATCH_SIZE, hidden_dim)
        if isinstance(sample_input, tuple):
            args_schema = tuple(BatchTensorDescriptor.from_tensor(arg, compression) for arg in sample_input)
        else:
            args_schema = (BatchTensorDescriptor.from_tensor(sample_input, compression),)
        print
        scheduler_cls = None

        # initialize experts
        experts = {}
        for expert_uid in expert_uids:
            expert = cls._make_expert(load_in_4bit,expert_cls,hidden_dim)
            cls._load_expert_weights(expert,expert_uid,checkpoint_dir,hugginface_rep)
            expert = expert.to(device)
            optimizer = optim_cls(expert.parameters()) if optim_cls is not None else None
            scheduler = scheduler_cls(optimizer) if scheduler_cls is not None else None
            if clip_grad_norm is not None:
                optimizer = ClippingWrapper(optimizer, clip_grad_norm)
            experts[expert_uid.uid] = ModuleBackend(
                name=expert_uid.uid,
                module=expert,
                device=device,
                args_schema=args_schema,
                optimizer=optimizer,
                scheduler=scheduler,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
            )
        torch.cuda.empty_cache()




        return cls(
            dht,
            experts,
            num_connection_handlers=num_handlers,
            device=device,
            stats_report_interval=stats_report_interval,
            update_period=update_period,
            expiration=expiration,
            start=start,
        )

    @classmethod
    def _make_expert(cls,load_in_4bit,expert_cls,hidden_dim):
        if load_in_4bit:
            expert = quantization(name_to_block[expert_cls](hidden_dim).to(dtype=torch.float16))
        else:
            expert = name_to_block[expert_cls](hidden_dim)
        return expert


    @classmethod
    def _load_expert_weights(cls, expert: nn.Module, expert_uid: ExpertUID, checkpoint_dir: Path, hugginface_rep: str):
        if checkpoint_dir and not hugginface_rep:
            load_expert(expert, expert_uid.uid, checkpoint_dir)
        elif not checkpoint_dir and hugginface_rep:
            load_expert_from_hf(expert, hugginface_rep, expert_uid.expert_id, expert_uid.layer_id)

    def run(self):
        """
        Starts Server in the current thread. Initializes dht if necessary, starts connection handlers,
        runs Runtime (self.runtime) to process incoming requests.
        """
        logger.info(f"Server started with {len(self.module_backends)} modules:")
        for expert_name, backend in self.module_backends.items():
            num_parameters = sum(p.numel() for p in backend.module.parameters() if p.requires_grad)
            logger.info(f"{expert_name}: {backend.module.__class__.__name__}, {num_parameters} parameters")

        if not self.dht.is_alive():
            self.dht.run_in_background(await_ready=True)

        if self.module_backends:
            self.dht_handler_thread.start()

        if self.checkpoint_saver is not None:
            self.checkpoint_saver.start()

        for handler in self.conn_handlers:
            handler.run_in_background()

        try:
            self.runtime.run()
        finally:
            self.shutdown()

    def run_in_background(self, await_ready=True, timeout=None):
        """
        Starts Server in a background thread. if await_ready, this method will wait until background server
        is ready to process incoming requests or for :timeout: seconds max.
        """
        self.start()
        if await_ready and not self.ready.wait(timeout=timeout):
            raise TimeoutError("Server didn't notify .ready in {timeout} seconds")

    @property
    def ready(self) -> mp.synchronize.Event:
        """
        An event (multiprocessing.Event) that is set when the server is ready to process requests.

        Example
        =======
        >>> server.start()
        >>> server.ready.wait(timeout=10)
        >>> print("Server ready" if server.ready.is_set() else "Server didn't start in 10 seconds")
        """
        return self.runtime.ready  # mp.Event that is true if self is ready to process batches

    def shutdown(self):
        """
        Gracefully terminate the server, process-safe.
        Please note that terminating server otherwise (e.g. by killing processes) may result in zombie processes.
        If you did already cause a zombie outbreak, your only option is to kill them with -9 (SIGKILL).
        """
        self.ready.clear()

        for handler in self.conn_handlers:
            handler.shutdown()
        logger.debug("Connection handlers terminated")

        if self.module_backends:
            self.dht_handler_thread.stop.set()
            self.dht_handler_thread.join()

        if self.checkpoint_saver is not None:
            self.checkpoint_saver.stop.set()
            self.checkpoint_saver.join()

        self.dht.shutdown()

        logger.debug(f"Shutting down runtime")
        self.runtime.shutdown()

        logger.info("Server shutdown successfully")


@contextmanager
def background_server(*args, shutdown_timeout=5, **kwargs) -> PeerInfo:
    """A context manager that creates server in a background , awaits .ready on entry and shuts down on exit"""
    pipe, runners_pipe = mp.Pipe(duplex=True)
    runner = mp.Process(target=_server_runner, args=(runners_pipe, *args), kwargs=kwargs)
    try:
        runner.start()
        # once the server is ready, runner will send us
        # either (False, exception) or (True, PeerInfo(dht_peer_id, dht_maddrs))
        start_ok, data = pipe.recv()
        if start_ok:
            yield data
            pipe.send("SHUTDOWN")  # on exit from context, send shutdown signal
        else:
            raise RuntimeError(f"Server failed to start: {data}")
    finally:
        runner.join(timeout=shutdown_timeout)
        if runner.is_alive():
            logger.info("Server failed to shutdown gracefully, terminating it the hard way...")
            runner.kill()
            logger.info("Server terminated")


def _server_runner(pipe, *args, **kwargs):
    try:
        server = Server.create(*args, start=True, **kwargs)
    except Exception as e:
        logger.exception(f"Encountered an exception when starting a server: {e}")
        pipe.send((False, f"{type(e).__name__} {e}"))
        return

    try:
        dht_maddrs = server.dht.get_visible_maddrs()
        pipe.send((True, PeerInfo(server.dht.peer_id, dht_maddrs)))
        pipe.recv()  # wait for shutdown signal

    finally:
        logger.info("Shutting down server...")
        server.shutdown()
        server.join()
        logger.info("Server shut down")


class ExpertUID:

    def __init__(self, uid : str, expert_id : int, layer_id: int):
        self.uid = uid
        self.expert_id = expert_id
        self.layer_id = layer_id

def _generate_uids(
    num_experts: int, num_layers: int, layers_start: int, expert_pattern: str, dht: Optional[DHT] = None, attempts_per_expert=10
) -> List[ExpertUID]:
    """
    Sample experts from a given pattern, remove duplicates.
    :param num_experts: sample this many unique expert uids
    :param expert_pattern: a string pattern or a list of expert uids,  example: myprefix.[0:32].[0:256]\
     means "sample random experts between myprefix.0.0 and myprefix.255.255;
    :param dht: if specified, uses this DHT to check that expert uids are not yet occupied by other peers
    :param attempts_per_expert: give up if unable to generate a new expert uid after this many attempts per uid
    :note: this method is not strictly process-safe. If several servers run it concurrently, they have
     a small chance of sampling duplicate expert uids.
    """

    def _generate_uid(layer_id, expert_id):
        return expert_pattern.format(expert_id=expert_id,layer_id=layer_id)
    

    # 1. sample uids
    new_uids = []
    for layer_id in range(layers_start, num_layers):
        for expert_id in range(num_experts):
            new_uid = _generate_uid(layer_id,expert_id)
            new_uids.append(ExpertUID(uid=new_uid,expert_id=expert_id,layer_id=layer_id))

        # 2. look into DHT (if given) and remove duplicates
    if dht is not None:
        found = get_experts(dht, [ui.uid for ui in new_uids])
        for f in found:
            if f is not None:
                raise NameError(f"{found} already exists")
    return new_uids
