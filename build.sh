pip install build  -i https://pypi.antfin-inc.com/simple/ 
pip install -r requirements.txt  -i https://pypi.antfin-inc.com/simple/ 
export COMPILE_CUSTOM_KERNELS=0 
export CPLUS_INCLUDE_PATH=/usr/lib/gcc/x86_64-openEuler-linux/12/../../../../include/c++/12:/usr/lib/gcc/x86_64-openEuler-linux/12/../../../../include/c++/12/x86_64-openEuler-linux:/usr/lib/gcc/x86_64-openEuler-linux/12/../../../../include/c++/12/backward:/usr/lib/gcc/x86_64-openEuler-linux/12/include:/usr/local/include:/usr/include
python -m build --no-isolation .
