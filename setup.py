from setuptools import setup

setup(
        name='lisl',
        version='0.1',
        url='https://github.com/funkelab/lisl',
        author='Riley White, Jan Funke',
        author_email='funkej@janelia.hhmi.org',
        license='MIT',
        packages=[
            'lisl',
            'lisl.models',
            'lisl.losses',
            'lisl.datasets',
            'lisl.train',
            'lisl.predict',
            'lisl.gp'
        ]
    )
