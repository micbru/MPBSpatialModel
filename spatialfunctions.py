import numpy as np
import scipy.fft
import scipy.optimize as opt
from scipy.special import betainc
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

##########################
# FUNCTIONAL FORMS
##########################
def fhill(m,n,g):
    return g**n/(m**n+g**n)

def ffull(m,k,p):
    return betainc(k,p+1,k/(k+m))

# Derivative of full function
def dffull(m,k,p):
    t1 = k/(k+m)
    t2 = m/(k+m)
    return - t1**k*t2**p/(k+m)/beta(k,p+1)

# Get the optimized function for every k
def getfopt(p,c,kmin=1e-2,kmax=1e2,returnk=False):
    '''
    This function returns an interpolated function that optimizes the beetle aggregation for each mean value of beetles by minimizing the survival function F. This works on a log range for small m, but then linear for larger m. We set a minimum and maximum for the aggregation k, so for very small number of beetles we set k=kmin, ie. we do not let the beetles aggregate infinitely, and for large numbers of beetles we set k=kmax. Outside of the interpolation range we set F=1 (for small m) and F=0 (for large m).
    If returnk = True, also return the k values themselves and the m values they were calculated at
    '''
    # Get the optimized k
    # Generate a list of k values for logspaced m between 0 and c
    mrange = np.logspace(-2,1) # Where this is 10^x
    # Make the range smaller to speed up this calculation.
    # First extend for each m in range of phi
    mrange = np.concatenate([mrange,np.arange(11,p+100)])
    # If c is still larger, interpolate further but in logspace rather than linear
    if c>p+100:
        mrange = np.concatenate([mrange,np.logspace(np.log10(p+100),np.log10(c))])
    kl = np.zeros(len(mrange)) # Now get ready to make the corresponding k list
    for i,m in enumerate(mrange):
        # Minimize the exponent of k to keep it positive, so bounds are in log space.
        sol = opt.minimize_scalar(lambda x: ffull(m,np.exp(x),p),
                                     bounds=[np.log(kmin),np.log(kmax)],method='bounded')
        kl[i] = sol.x
    # Take exponent to get true ks
    kl = np.exp(kl)
    # Now fix the beginning so that kl is the minimum until it isn't
    # Only do this if kl[0] is not kmin
    if np.abs(kl[0]-kmin)>0.1:
        # Find the first time it isn't kmax and set up to there as kmin
        first_index = np.argmax(np.abs(kl-kmax)>0.1)
        kl[0:first_index].fill(kmin)

    # Now create the optimized function
    # Extrapolate so that above range, everything dies. Below everything lives
    fopt = interp1d(mrange, ffull(mrange,kl,p), bounds_error=False, fill_value=(1.0,0.0))
    # Plot the k values if plot=true
    if returnk:
        return fopt, mrange, kl
    else:
        return fopt

##########################
# SPATIAL FUNCTIONS
##########################
# Laplace kernel
def laplace(x,al):
    '''Define the laplace kernal, defined as al/2 e^(-al|x|)'''
    k = 0.5*al*np.exp(-np.abs(x)*al)
    return k

def flaplace(xdx,dx,al):
    '''Get fft of laplace kernel on range x with spacing dx'''
    # fft of laplace kernel
    kdx = laplace(xdx,al)
    # Normalize
    kdx = kdx/np.sum(kdx*dx)
    # This shifts origin to the 0 index, which is what we want for fft
    ifkdx = scipy.fft.ifftshift(kdx)
    fkdx = scipy.fft.fft(ifkdx)#*dx # Extra dx factor appears here also
    # Actually it's easier to think of adding dx in the convolution step
    return fkdx

# Iteration
def iterate(xr,niter,f,x0,al,c,s,N,dx=1,**kwargs):
    '''Iterate for the number of steps equal to niter on the grid given by xr, with spacing dx (assumed to be 1)
    This function has a flexible response function f, and uses the laplace kernel with parameter alpha.
    c, s, and N are the beetle productivity, tree survival rate, and number of age classes.
    The args to be passed in are for the response function f.
    x0 gives the initial function distribution.
    '''

    # Get kernel fft
    fk = flaplace(xr,dx,al)
    
    # Set up return matrix
    xout = np.zeros([niter+1,N+3,len(xr)])
    
    # Set ut to initial function and get initial infested trees (since this can be calculated from other ICs)
    xt = x0
    ids = xt[N]>0
    mt = np.zeros(len(xr)) # Zero by default
    mt[ids] = xt[N+1,ids]/xt[N,ids] # Where not zero, calculate m
    
    # Get infested trees
    xt[N+2] = (1-f(mt,**kwargs))*xt[N]
    xout[0] = xt
    
    # Also set up an array for the new variables:
    xt1 = np.zeros([N+3,len(xr)])
    
    # And set up previous infestations
    # Could assume steady state, so infested trees were the same in the last two steps (as we do here)
    # Or could assume this is the first time we see beetles (ie. it1=it2=0, however then T!=1)
    it1 = xt[N+2]
    it2 = xt[N+2]
    
    # Loop through
    for i in np.arange(niter)+1:
        # Update all variables
        # Seedlings
        xt1[0] = (1-s)*np.sum(xt[:N],axis=0) + it2

        # Juveniles
        for j in np.arange(N-1):
            xt1[j+1] = s*xt[j]
        
        # Susceptibles
        xt1[N] = xt[N] - xt[N+2] + s*xt[N-1]
        
        # Beetles
        # Fourier transform It
        # First shift and take fft
        fit = scipy.fft.fft(scipy.fft.ifftshift(xt[N+2]))
        # Multipy with kernel to get beetles
        fkit = fk*fit*dx
        # Has to be shifted back
        # Can take abs, or real part and then set below zero to zero.
        bt1 = c*np.real(scipy.fft.fftshift(scipy.fft.ifft(fkit)))
        bt1[bt1<0] = 0
        xt1[N+1] = bt1
        
        # Infested trees
        ids = xt1[N]>0
        mt1 = np.zeros(len(xr)) # Zero by default
        mt1[ids] = xt1[N+1,ids]/xt1[N,ids] # Where not zero, calculate m
        xt1[N+2] = (1-f(mt1,**kwargs))*xt1[N]
    
        # Update previous infested trees it1 and it2
        it2 = it1
        it1 = xt[N+2]

        # Update xt
        xt = xt1.copy()
        
        # Save to array
        xout[i] = xt1
        
    return xout

############################
# METRICS
############################
# Get all metrics from a single parameter set
def get_outbreak_metrics(p,c,s,N,al,xr,xrexp,dx,niter,kmin=1e-2,kmax=1e2):
    '''
    This function takes in a single set of parameters and then returns three metrics for the outbreak:
    1. The speed of the wave.
    2. The period of the wave.
    3. The size of the outbreak as defined as the average maximum peak height over a single period, far enough
    out in time that we hopefully are close to asymptotic.
    The metrics are returned as 0 if the outbreak doesn't occur at all for the parameter set, which we check by
    first ensuring that there is a solution for 1-F = m/c, and second by ensuring the infested tree density 
    is above some cutoff at some point in time.
    '''
    # Before doing anything, we have to derive the optimum F function for these parameters
    fopt = getfopt(p,c,kmin=kmin,kmax=kmax)
    # Set size, speed, and period to 0, and only update if needed
    speed, period, size = 0,0,0
    # First, check to make sure 1-F = m/c exists. 
    # We only need to do this for phi/c > 0.75 (this is numerically true)
    if p/c>0.75:
        # This returns the value where (1-F)/m is maximized, which we can put in as a bound
        # Or returns an empty array, meaning no solution exists.
        mmax = outbreak_existence(fopt,c)
        if mmax.size>0:
            # If it returns a value, get the initial conditions
            x0 = get_x0even(xrexp,dx,fopt,c,s,N,bracket=[mmax,c+1])
        else:
            # If it is empty, set this array to empty which we will check later
            x0 = np.array([])
    # If the params are not that close to 0.75, we can just call this function directly
    else:
        x0 = get_x0even(xrexp,dx,fopt,c,s,N)
    
    # If after that we have an initial condition to run from, simulate the wave
    if x0.size>0:
        wave = iterate(xr,niter,fopt,x0,al,c,s,N,dx=dx)

        # Find the argument where the first transient outbreak occurs at x=1
        argmax = outbreak_occurs(wave,N,xrexp,dx)

        # Only calculate everything if the outbreak actually occurs
        if (argmax.size>0):
            # AND only if we have actually long enough data to calculate the size
            # (this assumes the period is N+3)
            if (argmax + (N+3)//2 + N+3 < niter):
                # Get speed
                # First check if the wave collides with the backward wave, only calculate the speed up to that arg
                argcol = collision_arg(wave,N,xrexp,dx)
                speed = getspeed(wave,argcol,N,xrexp,dx)

                # Get period at x = speed*(N+3) away from the origin, which assuming N+3 is one period away
                period = getperiod(wave,N,xrexp,dx,x=int(speed*(N+3)))
            
                # Now get the size
                size = getsize(wave,N,xrexp,dx,period=period,tstart=argmax+(period)//2)
    return speed, period, size

# Define tests for existence for the metrics

# Check if c*(1-F)/m > 1 and if so return the m where that is true
def outbreak_existence(fopt,c):
    '''
    Pass in the value of c and the interpolated maximum function and 
    look for the maximum value of c*(1-F)/m on the interval m e(1,c+1]
    If that value is >1, return the value of m at the maximum. If the value is <1, return an empty array.'''
    # Establish return array as empty
    ret = []
    # Minimize negative -c(1-F)/m
    Fm_min = opt.minimize_scalar(lambda x: -c*(1-fopt(x))/x,bounds=[1,c+1],method='bounded')
    # Check if it is >1
    if -Fm_min.fun>1:
        # If it is return the value where it is maximized to use as a bound
        ret.append(Fm_min.x)
    # Return the return array as an array
    return np.array(ret)

# Test when there is a peak at x=dx
def outbreak_occurs(wave,N,xrexp,dx,cutoff=1.05):
    '''
    This function tests if there is ever a peak in the wave at x=dx.
    It looks for the maximum argument at x=dx and then returns it if it is above 
    the cutoff multiplied by the initial size of the infestation 
    (ie. if it goes a certain amount above the initial setup)
    '''
    # This will be returned empty if there is no peak
    peakoccurs = []
    # Save the argument of the maximum
    maxatzero = np.argmax(wave[:,N+2,(2**xrexp)//dx+1])
    # If that is above the cutoff, then append the index and return it
    if wave[maxatzero,N+2,(2**xrexp)//dx+1] > cutoff*wave[0,N+2,0]:
        peakoccurs.append(maxatzero)
    return np.array(peakoccurs)

# Find when the wave collides with the backwards wave
def collision_arg(wave,N,xrexp,dx,threshold=1e-8):
    '''
    Find the time at which the wave collides with itself.
    By default use a threshold of 1e-8 (ie. numerical precision).
    Also, look at the number of beetles rather than the number of susceptible trees 
    to see when the dispersal step starts to intersect at all, rather than when the Allee threshold is overcome.
    '''
    # Check when the wave starts to interact with itself
    arg = np.where(wave[:,N+1,3*2**(xrexp-1)//dx]>threshold)[0]
    # If it never does, return the end time for this argument
    if arg.size==0:
        arg=wave.shape[0]-1
    else:
        arg=arg[0]
    return arg

# Period
def getperiod(wave,N,xrexp,dx,x=0):
    '''
    Get the length of time between peaks at position x (default 0).
    wave contains the wave information, the indices should be [time,age structure (N+1 is beetles),space]
    xrexp is the exponent for the spatial grid, which is assumed to go from -2**xrexp to 2**xrexp
    dx is the grid spacing
    Return the mean of the difference between peaks, but take the integer. It should not be fractional.
    '''
    # Get the peaks based on the height having to be 1 above the equilibrium height, which we get from the 0 time wave.
    # The peaks also have to be at least a little distance to avoid picking up internal peak structure.
    peaks, peak_info = find_peaks(wave[:,N+2,(2**xrexp+x)//dx],height=wave[0,N+2,0]*1.05,distance=0.6*N)
    periods = np.diff(peaks)
    # Check if all periods are equal
    flag = np.all(periods == periods.mean())
    if not flag:
        print("Periods are not all the same, ",periods)
        print("N=",N)
    # Return the median rather than the mean -- unless there is huge discrepancy in the printed statement, this should be ok
    return np.median(periods).astype(int)

def getsize(wave,N,xrexp,dx,period=np.nan,tstart=0,cutoff=1.0):
    '''
    This function returns the time averaged peak height of the infestation front. The average is over an entire period, starting from the second period. In other words, assuming a period T, the average is over the peak height at any location in space from times tstart to tstart + T. Note that rather than averaging directly over the peak height, it is easier to average the maximum of the wave overall. By default, tstart will be set to the period T. Note that this average, in tests, is the same for any period (ie. would be the same for 2T to 3T), except for the first from 0 to T as the wavefront initializes. For very slow waves, we need to start at a later time, which is why we have tstart as a variable. Cutoff will dictate how tall the peak has to be, which I will set to 1.0 x the equilibrium to start, the idea being that since we take the maximum it doesn't really matter how tall the peak is actually (we are not calculating the period here).
    wave contains the wave information, the indices should be [time,age structure (N+1 is beetles),space]
    period is the period T of the wave (which we assume is N+3, but is flexible)
    N is the number of generations before trees become susceptible
    xrexp is the exponent for the spatial grid, which is assumed to go from -2**xrexp to 2**xrexp
    dx is the grid spacing
    '''
    # If we do not define period, set it to N+3
    if period==np.nan:
        period=N+3
    # If the period is zero, the wave does not propogate, so the size is 0.
    elif period==0:
        return 0.
    # Set to start at time=period by default
    if tstart==0:
        tstart=period
    # Set the array for t where we will find the maximum peaks
    tarr = np.arange(tstart,tstart+period)
    peaks = np.zeros(period)
    for i,tt in enumerate(tarr):
        # Look for the maximum!
        # The +1 here is to prevent the wave at x=0 from dominating, since that doesn't make sense. 
        # This is only relevant for very slow waves.
        peaks[i] = np.max(wave[tt,N+2,(2**xrexp)//dx+1:(2**xrexp+2**(xrexp-1))//dx])
    return peaks.mean()

# Numerical speed
def getspeed(wave,ntest,N,xrexp,dx,xrange=0):
    '''
    wave contains the wave information, the indices should be [time,age structure (N+1 is beetles),space]
    ntest is the total amount of time the wave range for (or alternatively the length of time we'd like to calculate the speed over)
    xrexp is the exponent for the spatial grid, which is assumed to go from -2**xrexp to 2**xrexp
    dx is the grid spacing
    xrange is the range over which we look for the maximum of the wave (ie. where we look for the wave progressing). If it is 0, set it to 2**(xrexp-1) by default
    '''
    if xrange==0:
        xrange=2**(xrexp-1)
    # Get the leading args for the travelling wave
    leading_args = np.zeros([ntest+1])
    # Find max argument where the wave is advancing
    # Define threshold by average of wave to the left of center
    thresh = np.mean(wave[0,N+1,int(2**(xrexp-2)/dx):int(2**(xrexp-1)/dx)])
    for j in np.arange(ntest+1):
        # Look for maximum in some intermediate range. 
        # This may need fine tuning, so by default is 2**(xrexp-1) but can be changed
        temp = np.where(wave[j,N+1,int((2**xrexp-xrange)/dx):int((2**xrexp+xrange)/dx)]>thresh/2)[0]
        leading_args[j] = temp[-1]
    # Take the average over the last half of the time
    speed=np.mean(np.gradient(leading_args[ntest//2::]))*dx
    return speed

##########################
# OTHER FUNCTIONS
##########################

def get_eq(func,c,s,N,bracket=[],**kwargs):
    '''
    This function gets the equilibrium values for m,b,j, and s.
    The bracket determines if we are looking for the allee threshold or the upper equilibrium.
    With the hill function, this should be ms=c*(n-1)/n, but with other functions it is less clear what to use.
    c, s, and N are the beetle productivity, tree survival rate, and number of age classes.
    kwargs are passed to the function.
    It returns two values meq as a scalar, and an array of dimension N+2 with eq values of j -> b
    '''
    # If there is no bracket passed in, set it to a default.
    if not(any(bracket)):
        bracket=[c/2,c+1]
    ### Equilibrium numbers
    meq = opt.root_scalar(lambda x: x/c - 1 + func(x,**kwargs),bracket=bracket).root
    beq = meq*(1-s)*s**N/(c*(1-s)*s**N+meq-meq*s**N+2*meq*(1-s)*s**N)*c
    jeq = s**np.arange(N)*(beq/c)/s**N
    seq = beq/meq
    return meq, np.concatenate([jeq,[seq],[beq]])

def get_x0eq(xrexp,dx,func,c,s,N,bracket=[],**kwargs):
    '''
    This function is meant to make it easier to set up the travelling wave by setting everything to equilibrium values everywhere, but with beetles only on the LHS.
    It takes in xrexp, where the size of the grid is from -2**xrexp to 2**xrexp, and dx, the spacing of the grid
    c, s, and N are the beetle productivity, tree survival rate, and number of age classes.
    It also takes the function at which to calculate the equilibrium, with a default bracket from c/2 to c+1 since we want the upper equilibrium and not the Allee threshold.
    '''
    # If there is no bracket passed in, set it to a default.
    if not(any(bracket)):
        bracket=[c/2,c+1]
    
    meq,eqarr = get_eq(func,c,s,N,bracket=bracket,**kwargs)
    
    # Start everything at eq across all space, EXCEPT beetles.
    x0 = np.zeros([N+3,int(2**xrexp*2/dx)+1])
    for k in np.arange(N):
        x0[k].fill(eqarr[k])
    x0[N].fill(eqarr[N])

    # Now for beetles
    x0[N+1,0:int(2**xrexp/dx)] = eqarr[N+1]
    # And now make sure the total number of trees is correct
    x0[N,int(2**xrexp/dx)::] += 2*eqarr[N+1]/c
    return x0

def get_x0even(xrexp,dx,func,c,s,N,bracket=[],**kwargs):
    '''
    This function is meant to make it easier to set up the travelling wave by setting everything to equilibrium values everywhere on the left hand side, and then an even aged forest with S=1 on the RHS.
    It takes in xrexp, where the size of the grid is from -2**xrexp to 2**xrexp, and dx, the spacing of the grid
    c, s, and N are the beetle productivity, tree survival rate, and number of age classes.
    It also takes the function at which to calculate the equilibrium, with a default bracket from c/2 to c+1 since we want the upper equilibrium and not the Allee threshold.
    '''
    # If there is no bracket passed in, set it to a default.
    if not(any(bracket)):
        bracket=[0.9*c,c+1]
    
    meq,eqarr = get_eq(func,c,s,N,bracket=bracket,**kwargs)
    
    # Start everything at eq on the LHS
    x0 = np.zeros([N+3,int(2**xrexp*2/dx)+1])
    x0[:-1,:2**xrexp//dx] = np.repeat([eqarr],2**xrexp//dx,axis=0).T
    # And on RHS set all trees to 1
    x0[N,2**xrexp//dx::] = 1

    return x0