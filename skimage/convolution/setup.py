#!/usr/bin/env python

import os
from skimage._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('convolution', parent_package, top_path)
    config.add_data_dir('tests')

    cython(['ext.pyx'], working_path=base_path)
    args = ["-ffast-math "]
    config.add_extension('ext', sources=['ext.c'],
                         extra_compile_args=args,
                         include_dirs=[get_numpy_include_dirs()])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer = 'scikits-image Developers',
          author = 'scikits-image Developers',
          maintainer_email = 'scikits-image@googlegroups.com',
          description = 'Convolution',
          url = 'https://github.com/scikits-image/scikits-image',
          license = 'SciPy License (BSD Style)',
          **(configuration(top_path='').todict())
          )

