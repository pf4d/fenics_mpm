Installation
=======================

The main requirement of this software is FEniCS 2017.2.0.  I have detailed a procedure for installing from source `on my blog <https://pf4d.github.io/jekyll/update/2018/05/30/install-python.html>`_.

Latest Python packages and misc. dependencies::

  sudo pip install colored termcolor;

Install the program by editing your .bashrc file with::
  
  export PYTHONPATH="<PATH TO fenics_mpm>:$PYTHONPATH"

Test your installation py entering in an ``ipython`` terminal::

  from fenics_mpm import *

If this works, you will receive the message::

  Calling DOLFIN just-in-time (JIT) compiler, this may take some time.
  --- Instant: compiling ---

This is the software compiling the C++ backend, and is only performed once.  If a problem comes up with your installation, you can clear the compiled files by calling running the ``instant-clean`` command.  Finally, you can change the number of threads to use for your simulation by changing the environment variable ``OMP_NUM_THREADS`` to a number you desire; for example::

  export OMP_NUM_THREADS=2
