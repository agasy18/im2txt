from os import chdir, getcwd, path, mkdir
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
    def __init__(self, dir_path):
        self.dir = dir_path
        self.wd = None

    def __enter__(self):
        self.wd = getcwd()
        chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        chdir(self.wd)


is_gsutil_installed = False


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s %s/%s %s\r' % (bar, percents, '%', count, total, status))
    sys.stdout.flush()


def gs_download(url):
    print('Download:', url)
    d_name = path.basename(url)
    if path.isdir(d_name):
        print('Skip: {} dir already exists'.format(d_name))
        return d_name
    global is_gsutil_installed
    if not is_gsutil_installed:
        call_program(['gsutil', '--version'],
                     fail_info='can\'t find gsutil, please install `curl https://sdk.cloud.google.com | bash`')
        is_gsutil_installed = True
    mkdir(d_name)
    call_program(['gsutil', '-m', 'rsync', url, d_name])
    return d_name
