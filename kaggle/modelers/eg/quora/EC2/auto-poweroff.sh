#!/bin/bash

# @author: Eric Guo <guoanjie@gmail.com>
# @brief: automatically poweroff if all python processes have finished

cmd="ps -A -o comm | grep python"

while [ -n "$(eval $cmd)" ]; do
	sleep .1
done

sudo poweroff
