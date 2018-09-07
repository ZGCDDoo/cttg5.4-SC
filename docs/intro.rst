Introduction
================================

cttg stands for *Continuous time Tremblay group* 
and regroups two main Continuous-time quantum monte-carlo algorithms, namely CT-INT and CT-AUX.

Warning: Bad Docs
-----------------
The documentation has only been started very recently. It is thus full of language errors, probably wrong at certain places and of poor quality.
However, the quality will increase overtime.



Conventions
----------------

Comments
^^^^^^^^^

We use the convention "$" for the start of a shell command and "#" for commenting shell commands, etc.
    Ex:
        $ cd path/to/thing    # change to the path of thing.



Executable names
^^^^^^^^^^^^^^^^^
When not specified, the algorithm is CT-INT. if there is "aux" in the name, then CT-AUX.
If "sub" in name, then submatrix algorithm, else normal fast-update scheme.


Hamiltonian
^^^^^^^^^^^^

.. math::

   H = + \sum_{ij} t_{ij} + U \sum_{i} n_{i \uparrow} n_{i \downarrow}

Thus, for the cuprates, the convention of the program is

* t = -1.0
* t' = 0.3
* t'' = -0.2