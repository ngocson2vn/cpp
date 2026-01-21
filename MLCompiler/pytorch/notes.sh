torch_path=$(python3.11 -c 'import torch; print(torch.__path__[0])')
echo "torch_path = ${torch_path}"
touch ./solib_deps.log
for solib in $(find ${torch_path}/lib -type f -name "*.so")
do
  echo $solib >> solib_deps.log
  readelf -d $solib >> solib_deps.log
  echo >> solib_deps.log
done
echo ./solib_deps.log
