import errno
import logging
import os
import shutil
import tempfile

import munch
import requests
import sendfile
import six


def safe_rmtree(paths=None):
    if paths:
        for path in paths:
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                except (IOError, OSError) as e:
                    if e.errno == errno.ENOENT:
                        pass
                    else:
                        raise
                    

def safe_make_dir(a_dir, delete_existing=False):
    try:
        if delete_existing:
            safe_rmtree([a_dir])
        os.makedirs(a_dir)
    except (IOError, OSError) as e:
        if e.errno == errno.EEXIST and os.path.isdir(a_dir):
            pass
        else:
            raise