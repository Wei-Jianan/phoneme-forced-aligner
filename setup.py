from setuptools import setup, find_packages


if __name__ == '__main__':
    setup(
        name='htkaligner',
        version='1.0',
        packages=find_packages(),
        author='jianan wei',
        install_requires=['pypinyin', 'jieba'],
        package_data={
            'htkaligner': [
                'model/*',
                'model/*/*',

            ]
        }

    )