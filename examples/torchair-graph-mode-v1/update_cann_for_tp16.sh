cp /usr/local/Ascend/ascend-toolkit/8.1.RC1/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/libopmaster_ct.so /usr/local/Ascend/ascend-toolkit/8.1.RC1/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/libopmaster_ct.so.bak
cp /usr/local/Ascend/ascend-toolkit/8.1.RC1/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/libopmaster_rt2.0.so /usr/local/Ascend/ascend-toolkit/8.1.RC1/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/libopmaster_rt2.0.so.bak
cp /usr/local/Ascend/ascend-toolkit/8.1.RC1/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/liboptiling.so /usr/local/Ascend/ascend-toolkit/8.1.RC1/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/liboptiling.so.bak
cp /usr/local/Ascend/ascend-toolkit/8.1.RC1/opp/built-in/op_impl/ai_core/tbe/op_master_device/lib/Ascend-v7.7-libopmaster.so /usr/local/Ascend/ascend-toolkit/8.1.RC1/opp/built-in/op_impl/ai_core/tbe/op_master_device/lib/Ascend-v7.7-libopmaster.so.bak

cp /mnt/deepseek/liujiaxu/ifa_tiling.tgz .
tar -xvf /mnt/deepseek/liujiaxu/ifa_tiling.tgz
cp ./ifa_tiling/libopmaster_ct.so /usr/local/Ascend/ascend-toolkit/8.1.RC1/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/
cp ./ifa_tiling/libopmaster_rt2.0.so /usr/local/Ascend/ascend-toolkit/8.1.RC1/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/
cp ./ifa_tiling/liboptiling.so /usr/local/Ascend/ascend-toolkit/8.1.RC1/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/
cp ./ifa_tiling/Ascend-v7.7-libopmaster.so /usr/local/Ascend/ascend-toolkit/8.1.RC1/opp/built-in/op_impl/ai_core/tbe/op_master_device/lib/
rm -rf ./ifa_tiling
rm -rf ./ifa_tiling.tgz