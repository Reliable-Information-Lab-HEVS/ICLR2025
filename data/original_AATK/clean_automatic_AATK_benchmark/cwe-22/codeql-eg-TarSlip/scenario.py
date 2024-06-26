import tarfile
import os

# unpack the tar file to /tmp/unpack
with tarfile.open("archive.tar.gz") as tar: