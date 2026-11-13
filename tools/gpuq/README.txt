gq -- the GPU job queue of this project (2 x RTX 5090, 32607 MiB each).  ALL GPU jobs go through it.
GQ=$RESCOPE_ROOT/02_research/tools/gpuq/gq          (code: gq.py, state: state/)

COMMANDS
  $GQ run --mem MiB [--gpu any|0|1] [--name TAG] [--prio INT] [--no-retry] -- command ...
      Blocks until the job may start, then runs the command with CUDA_VISIBLE_DEVICES = the chosen card (the
      job sees it as cuda:0).  stdout, stderr and the exit code are the command's (128 + n if signal n ended
      it).  gq adds three lines on stderr: queued / started (card, seconds waited) / exit.
      --mem   what the job may take on its card as nvidia-smi counts it (torch memory + ~500 MiB CUDA context)
      --prio  higher is served first (default 0)         --name  tag shown by gq status
  $GQ status                    one table: per card total / used / free / reserved by queue jobs / foreign / FITS
                                (largest --mem that could start now); running jobs (reserved and actual MiB,
                                runtime); waiting jobs (need, wait time, why they have not started)
  $GQ wait-idle [--timeout S]   returns 0 when nobody's queue job is running or waiting, 1 after S seconds

ADMISSION RULE (a waiting job looks every 15 s; one lock around: read nvidia-smi, choose the card, reserve)
  1. Fit:    total - max(used, queue reservations + foreign) >= mem + 700 MiB   (foreign: memory of other processes)
  2. Busy:   no start on a card with a foreign process younger than 120 s or foreign growth > 300 MiB in 10 s.
  3. Order:  priority, then ticket time.  A later job may start if it fits now (backfill).
  4. Aging:  after a 15 min wait, later tickets of the same or lower priority stay off cards it could use without them.
  5. Card:   --gpu if given, else the fitting card with LESS free memory.  The reservation ends with the command.

EXAMPLES (copy and paste)
  GQ=$RESCOPE_ROOT/02_research/tools/gpuq/gq; mkdir -p logs
  nohup $GQ run --mem 10500 --name e1_qwen -- python run.py --model qwen ... > logs/e1.out 2>&1 &
  nohup $GQ run --mem 17000 --name e1_llava --prio 5 -- python run.py --model llava ... > logs/e1_llava.out 2>&1 &
  $GQ status
Start all your jobs at once (a loop of such lines); the queue orders them.  `wait` waits for your own background
jobs.  `kill <pid of a gq run>` stops that job: its command gets the same signal, the reservation is removed.
The command runs as given, without a shell: for `cd dir && ...` or pipes write  -- bash -c '...'

MEMORY FIGURES (use them unless you have measured your job: gq status, column ACTUAL)
  Qwen2.5-VL-3B   --mem 10500          LLaVA-1.5-7B   --mem 17000
  The job must also cap itself, so that it cannot grow into its neighbours.  First lines of the job:
      import os, torch
      cap = (int(os.environ["GQ_MEM"]) - 600) * 2**20           # GQ_MEM = the --mem of this job, set by gq run
      torch.cuda.set_per_process_memory_fraction(cap / torch.cuda.get_device_properties(0).total_memory, 0)
  A job that takes more than its --mem is marked "!" in gq status: give it a larger --mem next time.

RULES
  * No other launcher.  No gpu_wait.sh / gpu_run.sh / gpu_any.sh / gpu_fill.sh / gpu_big.sh / gpu_launch.sh, no
    polling loop of your own on nvidia-smi, no bare `CUDA_VISIBLE_DEVICES=N python ...`.  A process outside the
    queue is foreign: the queue sees only what it has taken so far, and the next job that loads next to it
    runs out of memory.  Do not set CUDA_VISIBLE_DEVICES yourself and do not wait for a free card yourself.
  * One gq run around each GPU process, or around a script whose GPU steps run one after another (--mem = the
    largest step).  Steps as separate jobs:  $GQ run ... -- python step1.py && $GQ run ... -- python step2.py
  * A CUDA out-of-memory failure in the first 120 s of the command is queued again once, 60 s later, instead of
    being returned (--no-retry: returned at once).  A second failure is returned.  Make jobs resumable.
  * gq never kills or changes other processes.  Do not delete or edit the files in state/.
  * kill -9 of a gq run frees its reservation at the next look, but only the direct child gets a SIGTERM:
    processes below it may live on as foreign ones.  Use plain kill.

TESTS (python = $RESCOPE_PYTHON, run in this directory)
  python tests/test_admission.py     the rule on injected nvidia-smi readings (no GPU, 2 s)
  python tests/test_queue.py         the whole tool with a fake nvidia-smi and fake jobs (no GPU, 2.5 min)
  python tests/test_gpu.py           torch dummy jobs on a real card through the live queue (6 min; --dry: no GPU)
