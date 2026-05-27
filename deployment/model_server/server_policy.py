# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import logging
import socket
import argparse
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework
import torch, os


def main(args) -> None:
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()

    vla = baseframework.from_pretrained( # TODO should auto detect framework from model path
        args.ckpt_path,
    )

    device = torch.device(f"cuda:{str(args.cuda)}")

    if args.use_bf16: # False
        vla = vla.to(torch.bfloat16)
    vla = vla.to(device).eval()

    if args.per_task_dir:
        vla.load_kf_per_task(args.per_task_dir)
        logging.info("Per-task KF enabled: %s", args.per_task_dir)
    elif args.lds_path:
        vla.load_kf(args.lds_path, q_noise=args.kf_q, r_noise=args.kf_r)
        q_str = f"{args.kf_q:.3f}" if args.kf_q is not None else "EM"
        r_str = f"{args.kf_r:.3f}" if args.kf_r is not None else "EM"
        logging.info("KF enabled: %s  q=%s  r=%s", args.lds_path, q_str, r_str)
        if args.adaptive:
            vla.enable_adaptive_placement(
                placement_alpha=args.placement_alpha,
                opening_threshold=args.opening_threshold,
            )
        if args.adaptive_r:
            vla.enable_adaptive_r(gamma=args.ar_gamma,
                                  clip_lo=args.ar_clip_lo, clip_hi=args.ar_clip_hi)
        if args.approach_aware:
            vla.enable_approach_aware(R_scale=args.app_R_scale,
                                      slow_speed=args.app_slow_speed,
                                      gripper_close=args.app_gripper_close)
    elif args.ema_alpha is not None:
        vla.load_ema(args.ema_alpha)
        logging.info("EMA enabled: alpha=%.3f", args.ema_alpha)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # start websocket server
    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        metadata={"env": "simpler_env"},
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--cuda", default=0)
    parser.add_argument("--lds_path", type=str, default=None, help="Path to LDS .npz for KF filtering")
    parser.add_argument("--per_task_dir", type=str, default=None,
                        help="Dir with per-task LDS (lds_em_task*.npz + task_index.json); auto-selects by instruction")
    parser.add_argument("--kf_q", type=float, default=None,
                        help="Process noise; if None, use EM-learned q_em from LDS ckpt")
    parser.add_argument("--kf_r", type=float, default=None,
                        help="Observation noise; if None, use EM-learned r_em from LDS ckpt")
    parser.add_argument("--adaptive", action="store_true",
                        help="Enable placement-aware KF (reduce smoothing when gripper opens)")
    parser.add_argument("--placement_alpha", type=float, default=0.2,
                        help="KF blend during gripper-opening (0=raw, 1=full KF)")
    parser.add_argument("--opening_threshold", type=float, default=0.001,
                        help="Gripper qpos delta to trigger placement mode")
    parser.add_argument("--adaptive_r", action="store_true",
                        help="Per-step adaptive R from token spread (true adaptive KF)")
    parser.add_argument("--ar_gamma", type=float, default=1.0)
    parser.add_argument("--ar_clip_lo", type=float, default=0.5)
    parser.add_argument("--ar_clip_hi", type=float, default=2.0)
    parser.add_argument("--approach_aware", action="store_true",
                        help="Weaken KF during slow precision motion (holding object)")
    parser.add_argument("--app_R_scale",      type=float, default=5.0)
    parser.add_argument("--app_slow_speed",   type=float, default=0.01)
    parser.add_argument("--app_gripper_close",type=float, default=0.03)
    parser.add_argument("--ema_alpha", type=float, default=None, help="EMA smoothing alpha (0~1); overridden by --lds_path")
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10091))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10091 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
