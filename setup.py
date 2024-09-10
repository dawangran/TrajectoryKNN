from setuptools import setup 
from setuptools import find_packages

setup(name='TrajectoryKNN',
      version='0.0.1',
      description='Look for genes that express something similar',
      author='dawn',
      author_email='605547565@qq.com',
      requires= ['pandas','scanpy','numpy','anndata','sklearn','pyecharts'], # 定义依赖哪些模块
      packages=find_packages(),  # 系统自动从当前目录开始找包
      license="apache 3.0"
      )
