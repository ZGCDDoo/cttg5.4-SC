The params file
#############################

Fast-Update Scheme
=========================

Main Paremeters
-----------------------
The following parameters are the ones that the user has to change and need to understand. There is a bit of unconsistency, to be corrected
(some are lower case, while other are upper case.)



    modelType
        the model to run the simulations, can be one of the following:
        * "SIAM_Square"
        * "Square2x2"
        * "Square4x4"
        * "Triangle2x2"

    solver
        the solver to use, can be one of the following:

        =====   =========  
        cttg    cttg_sub 
        =====   =========
        "Int"   "IntSub"
        "Aux"   "IntAux" 
        =====   =========
    
    SEED
        the seed for the random number generator.

    beta
        inverse temperature

    EGreen
        the cutoff of matsubara frequencies in energy for the measurement of the green fucntions, 100 is fine.

    NTAU
        the time discretization for G(tau) and for binning measurements. Normally
        a value of 1000 is sufficient, but, for low temperatures and big EGreen,
        a higher value is neccessary to get unbiaised results. I recommend NTAU ~ beta 150.
        If you so wish, I take a minimum value of NTau = beta*125 in the code. I thus take
        only NTau if it is bigger than this minimum value.
        
        =====   =====  
        beta    NTAU 
        =====   =====
        10      1500
        50      7500 
        =====   =====


    UPDATESMEAS 
        The numbre of Updates proposed bewteen each measurement.
        This value should be approximately equal to the average expansion order. 
        Updates proposed bewteen measurements = UPDATESMEAS. I recommend UPD < k.
        
        =====  =====  
        k       UPD 
        =====  =====
        200     100
        500     250
        800     300
        1000    350
        1500    400  
        =====  =====
        

    THERMALIZATION_TIME
        the time in minutes for which each processor will thermalize. It is difficult to give a good
        optimal value. I would say, ~10% of the measurement time.

    MEASUREMENT_TIME
        The time in minutes each processor measures. For Mp2, 1 node:

        =====  ==================  
        k       MT 
        =====  ==================
        200     5
        500     20 try submatrix
        800     35 try submatrix
        1000    50 try submatrix
        1500    90 try submatrix
        =====  ==================

    WEIGHTSR, WEIGHTSI
        The real and imaginary part of w in the following:
        hyb_{n+1} = w*hyb_n + (1-w)*hyb_{n+1}


    N_T_INV
        The number of translational invariance measurements to take for ONE given configuration. 5 is a good value. Reduces noise for filling and docc.
        Do not put a too big value.

    ESelfCon
        The cut off in energy to do the SelfConsistency

    n
        If this parameter is in the params file, than the program will change the chemical potential to attain the given value.

    S
        If n is given in the params file, then S should also be given. It controls the change of chemical potentiel
        according to a newton method. ~1 is ok.

Implementation detailed Parameters
-----------------------------------

These parameters can be left at their current value, i.e they are implementation details.
Make sure you understand what you are doing before changing them.



    CLEANUPDATE
        Specifies when to perform a clean update. Ex, if =100, than at each
        100 measures, a cleanupdate will be performed. 100 is a good number.
        does not substantially influence the simulation, except if this number is to low or to high.
        
    K
        The value of the K parameter of CT-Aux. Influences the acceptance rate and the expansion order
        1 seems a reasanable value

    delta
        The value of the delta parameter of CT-INT. Influences the acceptance rate and the expansion order
        ~0.01 seems a reasanable value

    THERM_FROM_CONFIG
        if true, there will be no thermalization, the last saved configuartion will be loaded
        and the measurements will start. Not really tested yet. So the default is false.
        André-Marie argues that it is better to thermalize each time.
    



Submatrix Update Scheme
=========================

    KMAX_UPD 
        the maximum number of updates proposed for each internal iteration (implementation details).
        This parameter should be ~100 on MP2, ~125 on Graham, and has a different optimal value for different architectures.