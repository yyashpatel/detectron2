import os
import sys
import glob
import argparse

"""
Argument Parser
"""
def parseArgs():
    print("Parsing arguments ...")

    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    
    parser.add_argument("--config-file", default=False, metavar="FILE", help="path to config file",)

    parser.add_argument('--input_data',  type=str,  default=False,  help='')

    parser.add_argument( "--output",type=str, help="A file or directory to save output visualizations.",)

    parser.add_argument("--confidence-threshold", type=float, default=0.5,help="Minimum score for instance predictions to be shown",)

    parser.add_argument("--score_threshold", type=float, default=0.5,help="Minimum score for instance predictions to be shown",)

    parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs", default=[],nargs=argparse.REMAINDER,)
    
    parser.add_argument("--resume",action="store_true", help="Whether to attempt to resume from the checkpoint directory."" DefaultTrainer.resume_or_load()",)
    
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)")

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    return parser.parse_args()



def checkFolderPaths(folder_paths):
  for path in folder_paths:
    if not os.path.exists(path):
      os.makedirs(path)
      print("Creating folder", path, "...")