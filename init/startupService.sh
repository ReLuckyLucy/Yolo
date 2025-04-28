#!/bin/bash

# 显示motd
cat /etc/motd

# 启动SSH服务
/etc/init.d/ssh start

# 保持容器运行
tail -f /dev/null 