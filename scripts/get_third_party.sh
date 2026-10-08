#!/usr/bin/env bash
# Reference implementations of the competitors, at the commits used in the paper (not redistributed here).
# They are only needed for the corresponding baselines; ours needs none of them.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
get() {  # name url commit
  [ -d "third_party/$1" ] || git clone -q "$2" "third_party/$1"
  git -C "third_party/$1" checkout -q "$3"
  echo "third_party/$1 @ $3"
}
get BiSeNet         https://github.com/CoinCheung/BiSeNet                        6b4b67a  # BiSeNetV2 (MIT)
get DNANet          https://github.com/YeRen123455/Infrared-Small-Target-Detection 21584be  # DNANet (MIT)
get MSHNet          https://github.com/ying-fu/MSHNet                            46cdfd4  # MSHNet
get MagNet          https://github.com/VinAIResearch/MagNet                      7f5b876  # MagNet (AGPL-3.0)
get SuperSimpleNet  https://github.com/blaz-r/SuperSimpleNet                     98ab4d5  # SuperSimpleNet (MIT)
