"""Download nflverse play-by-play (about 160 MB for 2019 to 2026)."""
import sys

from gamescript.data import download

if __name__ == "__main__":
    download(force="--force" in sys.argv)
