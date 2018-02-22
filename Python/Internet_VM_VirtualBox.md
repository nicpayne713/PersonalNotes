## Oracle Virtual Machine VirtualBox
* Ubuntu 16.04 iso
* MB ~ 8gb
* Hard Drive space ~ 25 gb

File -> Preferences -> Proxy -> http://proxy.cat.com:80
Network settings set http/https/socks proxies

set 2 network adapters from VM launch gui
* NAT
* Host-only

## Python
https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04

* sudo apt-get update
* sudo apt install curl
* cd /tmp
* curl -0 https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
* **Might just need to go to website and download since curl can timeout**
* bash Anaconda3-5.0.1-Linux-x86_64.sh

## PyTorch
* conda install pytorch-cpu torchvision -c pytorch

## PyCharm

* sudo snap install pycharm-community --classic
* if this doesn't work then download from jetbrains.com
  * archive manager unpack to wherever you want
  * subdirectory is bin/
    * bash pycharm.sh to start *pycharm*
* **LOCK PYCHARM TO TASKBAR TO NOT NEED BASH CALL EACH TIME**
