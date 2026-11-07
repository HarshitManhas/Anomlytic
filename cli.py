import os
import warnings

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
warnings.filterwarnings("ignore")

from anomlytic.cli import main

if __name__ == "__main__":
    main()
