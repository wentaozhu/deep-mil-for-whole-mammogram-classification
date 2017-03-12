from distutils.core import setup

setup(name='convnetskeras',
      version='0.1',
      description='Pre-trained convnets in Keras',
      author='Leonard Blier',
      author_email='leonard.blier@ens.fr',
      packages=['convnetskeras'],
      package_dir={'convnetskeras':'convnetskeras'},
      package_data={'convnetskeras':["data/*"]},
      long_description=open('README.md').read(),
     )

