from os import chdir, getcwd, path, mkdir, rename, makedirs
from subprocess import call
from sys import exit


import sys

DEBUG = False


def call_program(args, shell=False, exit_on_error=True, fail_info=''):
    """
call external program
Examples
    call(['mkdir', '-p', dir]) # create dir path if not exist
    Args:
        :param args: array of arguments (first item is program name)
        :param shell: run on system shell
        :param exit_on_error: exit if program returns not 0
        :param fail_info: log additional fail info
    """
    if DEBUG:
        print('Calling `{}`, working dir: {}'.format(' '.join(args), getcwd()))
    res = call(args, shell=shell)
    if res != 0 and exit_on_error:
        exit('Failed with {} `{}`, working dir: {} {}'.format(res, ' '.join(args), getcwd(), fail_info))
    return res


class working_dir(object):
    def __init__(self, dir_path, create_if_not_exists=False):
        self.create_if_not_exists = create_if_not_exists
        self.dir = dir_path
        self.wd = None

    def __enter__(self):
        self.wd = getcwd()
        path.isdir(self.dir) or makedirs(self.dir, exist_ok=True)
        chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        chdir(self.wd)


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(bar_len * (count % total) / total)
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s %s/%s %s\r' % (bar,
                                               percents if percents < 100 else '?',
                                               '%', count, total if total > count else '?',
                                               status))
    sys.stdout.flush()


_is_gsutil_installed = False


def gs_download(url, d_name=None):
    print('Download:', url)
    d_name = d_name or path.basename(url)
    if path.isdir(d_name):
        print('Skip: {} dir already exists'.format(d_name))
        return d_name
    global _is_gsutil_installed
    if not _is_gsutil_installed:
        call_program(['gsutil', '--version'],
                     fail_info='can\'t find gsutil, please install `curl https://sdk.cloud.google.com | bash`')
        _is_gsutil_installed = True
    if not path.exists(d_name + "_tmp"):
        mkdir(d_name + "_tmp")
    call_program(['gsutil', '-m', 'rsync', url, d_name + "_tmp"])
    rename(d_name + "_tmp", d_name)
    return d_name


