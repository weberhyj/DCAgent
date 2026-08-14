#!/usr/bin/env bash

NODE_BIN=/usr/local/bin
YARN_BIN=$HOME/.yarn/bin
FAST_BIN=$HOME/.local/bin

if [[ ":$PATH:" == *":$NODE_BIN:"* ]]; then
    node -v
else
    PATH=$NODE_BIN:$PATH
fi
if [[ ":$PATH:" == *":$YARN_BIN:"* ]]; then
    yarn -v
else
    PATH=$YARN_BIN:$PATH
fi
if [[ ":$PATH:" == *":$FAST_BIN:"* ]]; then
    fast --version
else
    PATH=$FAST_BIN:$PATH
fi

set -x
set -e

git pull

echo Going to build and deploy admin page ...
cd admin-frontend/
./update_deploy.sh

echo Deploying frontend ...
cd ../frontend/
./update_deploy.sh

echo Install deps and restart backend ...
cd ../backend/
fast pypi --reverse --quiet
fast deps --uv
sudo supervisorctl restart all
sudo supervisorctl status
git checkout -- uv.lock
echo Done.
