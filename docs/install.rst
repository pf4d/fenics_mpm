Installation
=======================

FEniCS 2016.2.0::

  sudo add-apt-repository ppa:fenics-packages/fenics;
  sudo apt-get update;
  sudo apt-get install fenics;
  sudo apt-get dist-upgrade;

Latest Python packages and misc. dependencies::

  sudo pip install colored termcolor;

Install the program by editing your .bashrc file with::
  
  export PYTHONPATH="<PATH TO fenics_mpm>:$PYTHONPATH"

Test your installation py entering in an ``ipython`` terminal::

  from fenics_mpm import *



