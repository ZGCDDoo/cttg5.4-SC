==========================================================================
 CTTG5.3 : Continuous time Tremblay Group
==========================================================================

:Authors: Charles-David Hébert, Maxime Charlebois, Patrick Sémon 
:Date: $Date: 2018-06-21 $
:Revision: $Revision: 5.3.0 $
:Description: Description

Naming conventions
-------------------
See this site:
    https://docs.microsoft.com/en-us/dotnet/standard/design-guidelines/naming-guidelines
 

General guidelines before pull requests
----------------------------------------

Build
^^^^^^^^^^^^^^^^^^^^^^
* Must build with clang3.8-clang6.0 for serial mode, no warnings, except for external libraries
* Must build with g++5.4-g++8.0 for MPI mode, no warings, except for external libraries
* Must pass all tests
* Must repredouce the results in "test/Simulations"


Naming conventions
^^^^^^^^^^^^^^^^^^^
* Must respect the naming conventions


Formatting
^^^^^^^^^^^^^^^^
* Formatted according to Visual Studio standard, for exemple in visual studio code, put the folling preferences:
* "C_Cpp.clang_format_style": "Visual Studio"
* "editor.formatOnSave": true



    