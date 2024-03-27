#!/bin/sh

TIME_BASED_ID=$(date +%Y%m%d%H%M%S)

FILENAME=profile_log_${TIME_BASED_ID}.txt

echo "Results with yjit" >$FILENAME
rbenv local 3.2.3
/usr/bin/time -h -l -a -o $FILENAME bundle exec ruby --yjit --yjit-exec-mem-size=512 ./examples/ppo_gridworld.rb

echo "Results without yjit" >>$FILENAME
/usr/bin/time -h -l -a -o $FILENAME bundle exec ruby ./examples/ppo_gridworld.rb
