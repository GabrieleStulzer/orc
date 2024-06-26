import numpy as np
from numpy import nan
from numpy.linalg import inv, norm
from math import sqrt
import matplotlib.pyplot as plt
import orc.utils.plot_utils as plut
import time, sys
from orc.utils.robot_loaders import loadUR
from orc.utils.robot_wrapper import RobotWrapper
from orc.utils.robot_simulator import RobotSimulator
import A1_conf as conf

print("".center(conf.LINE_WIDTH,'#'))
print(" Manipulator: Impedence Control vs. Operational Space Control vs. Inverse Kinematics + Inverse Dynamics ".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH,'#'), '\n')

PLOT_TORQUES = 0
PLOT_EE_POS = 0
PLOT_EE_VEL = 0
PLOT_EE_ACC = 0

r = loadUR()
robot = RobotWrapper(r.model, r.collision_model, r.visual_model)

if conf.TRACK_TRAJ:
    tests = []
    
    tests += [{'controller': 'OSC', 'kp': 100,  'frequency': np.array([1.0, 1.0, 0.3]), 'friction': 2}]
    tests += [{'controller': 'IC',  'kp': 100,  'frequency': np.array([1.0, 1.0, 0.3]), 'friction': 2}]

    tests += [{'controller': 'OSC', 'kp': 100,  'frequency': 3*np.array([1.0, 1.0, 0.3]), 'friction': 2}]
    tests += [{'controller': 'IC',  'kp': 100,  'frequency': 3*np.array([1.0, 1.0, 0.3]), 'friction': 2}]
   
    # Tweak proportional gain
    tests += [{'controller': 'OSC', 'kp': 500,  'frequency': np.array([1.0, 1.0, 0.3]), 'friction': 2}]
    tests += [{'controller': 'IC',  'kp': 500,  'frequency': np.array([1.0, 1.0, 0.3]), 'friction': 2}]

    # Tweak proportional gain at higher frequencies
    tests += [{'controller': 'OSC', 'kp': 500,  'frequency': 3*np.array([1.0, 1.0, 0.3]), 'friction': 2}]
    tests += [{'controller': 'IC',  'kp': 500,  'frequency': 3*np.array([1.0, 1.0, 0.3]), 'friction': 2}]
else:
    tests = []

    # tests += [{'controller': 'IC_O_simpl',  'kp': 250,  'frequency': np.array([0.0, 0.0, 0.0]),  'friction': 0}]        
    # tests += [{'controller': 'IC_O_simpl_post',  'kp': 250,  'frequency': np.array([0.0, 0.0, 0.0]),'friction': 0}]    
    # tests += [{'controller': 'IC_O',  'kp': 250, 'frequency': np.array([0.0, 0.0, 0.0]), 'friction': 0}]              
    # tests += [{'controller': 'IC_O_post',  'kp': 250, 'frequency': np.array([0.0, 0.0, 0.0]), 'friction': 0}]         

    tests += [{'controller': 'IC_O_simpl',  'kp': 250,'frequency': np.array([0.0, 0.0, 0.0]), 'friction': 2}]        
    tests += [{'controller': 'IC_O_simpl_post',  'kp': 250,'frequency': np.array([0.0, 0.0, 0.0]), 'friction': 2}]       
    tests += [{'controller': 'IC_O',  'kp': 250,'frequency': np.array([0.0, 0.0, 0.0]), 'friction': 2}]               
    tests += [{'controller': 'IC_O_post',  'kp': 250,'frequency': np.array([0.0, 0.0, 0.0]), 'friction': 2}]       



plt.tight_layout(rect=[0, 0, 1, 0.95])
frame_id = robot.model.getFrameId(conf.frame_name)

simu = RobotSimulator(conf, robot)

if conf.TRACK_TRAJ:
    tracking_err_osc = []          # list to contain the tracking error of OSC
    tracking_err_ic  = []          # list to contain the tracking error of IC
else:
    stab_err_ic_O_simpl = []       # list to contain the tracking error of IC_O_simpl
    stab_err_ic_O_simpl_post = []  # list to contain the tracking error of IC_O_simpl_post
    stab_err_ic_O = []             # list to contain the tracking error of IC_O
    stab_err_ic_O_post = []        # list to contain the tracking error of IC_O_post

for (test_id, test) in  enumerate(tests):
    if conf.TRACK_TRAJ:
        description = str(test_id)+' Controller '+test['controller']+' kp='+\
                  str(test['kp'])+' frequency=[{},{},{}]'.format(test['frequency'][0],test['frequency'][1],test['frequency'][2])+' friction='+str(test['friction'])
    else:
        description = str(test_id)+' Controller '+test['controller']+' kp='+str(test['kp'])+' friction='+str(test['friction'])
    print(description)

    kp = test['kp']             # proportional gain of tracking task
    kd = 2*np.sqrt(kp)          # derivative gain of tracking task
    kp_j = 20.0                 # proportional gain of end effector task
    kd_j = 2*sqrt(kp_j)         # derivative gain of end effector task

    freq = test['frequency']

    if test['friction'] == 0:
        simu.simulate_coulomb_friction = 0
    else:
        simu.simulate_coulomb_friction = 1

    simu.tau_coulomb_max = test['friction']*np.ones(6)  # To overwrite the tau_coulomb_max in conf
    simu.init(conf.q0)                                  # initialize simulation state
    
    nx, ndx = 3, 3                              # size of x and its time derivative
    N = int(conf.T_SIMULATION/conf.dt)          # number of time steps
    tau     = np.empty((robot.na, N))*nan       # joint torques
    tau_c   = np.empty((robot.na, N))*nan       # joint Coulomb torques
    q       = np.empty((robot.nq, N+1))*nan     # joint angles
    v       = np.empty((robot.nv, N+1))*nan     # joint velocities
    dv      = np.empty((robot.nv, N+1))*nan     # joint accelerations
    x       = np.empty((nx,  N))*nan            # end-effector position
    dx      = np.empty((ndx, N))*nan            # end-effector velocity
    ddx     = np.empty((ndx, N))*nan            # end effector acceleration
    x_ref   = np.empty((nx,  N))*nan            # end-effector reference position
    dx_ref  = np.empty((ndx, N))*nan            # end-effector reference velocity
    ddx_ref = np.empty((ndx, N))*nan            # end-effector reference acceleration
    ddx_des = np.empty((ndx, N))*nan            # end-effector desired acceleration

    t = 0.0
    PRINT_N = int(conf.PRINT_T/conf.dt)
    
    for i in range(0, N):
        time_start = time.time()
        
        # set reference trajectory
        if conf.TRACK_TRAJ:
            # set reference trajectory
            two_pi_f             = 2*np.pi*freq   # frequency (time 2 PI)
            two_pi_f_amp         = np.multiply(two_pi_f, conf.amp)
            two_pi_f_squared_amp = np.multiply(two_pi_f, two_pi_f_amp)
            x_ref[:,i]  = conf.x0 +  conf.amp*np.sin(two_pi_f*t + conf.phi)
            dx_ref[:,i]  = two_pi_f_amp * np.cos(two_pi_f*t + conf.phi)
            ddx_ref[:,i] = - two_pi_f_squared_amp * np.sin(two_pi_f*t + conf.phi)
        else:
            # stabilize the point np.array([0.2,0.2,0.2])
            x_ref[:,i]   = conf.x_ref_O
            dx_ref[:,i]  = 0.0
            ddx_ref[:,i] = 0.0
            
        # read current state from simulator
        v[:,i] = simu.v
        q[:,i] = simu.q
        
        # compute mass matrix M, bias terms h, gravity terms g
        robot.computeAllTerms(q[:,i], v[:,i])
        M = robot.mass(q[:,i], False)
        h = robot.nle(q[:,i], v[:,i], False)
        g = robot.gravity(q[:,i])
        
        J6 = robot.frameJacobian(q[:,i], frame_id, False)
        J = J6[:3,:]                                                                    # take first 3 rows of J6
        H = robot.framePlacement(q[:,i], frame_id, False)
        x[:,i] = H.translation                                                          # take the 3d position of the end-effector
        v_frame = robot.frameVelocity(q[:,i], v[:,i], frame_id, False)
        dx[:,i] = v_frame.linear                                                        # take linear part of 6d velocity
        #    dx[:,i] = J.dot(v[:,i])
        dJdq = robot.frameAcceleration(q[:,i], v[:,i], None, frame_id, False).linear
        
        # implement your control law here
        ddx_fb = kp * (x_ref[:,i] - x[:,i]) + kd*(dx_ref[:,i] - dx[:,i])
        ddx_des[:,i] = ddx_ref[:,i] + ddx_fb
        Minv = inv(M)
        J_Minv = J.dot(Minv)
        Lambda = inv(J_Minv.dot(J.T))
        mu = Lambda.dot(J_Minv.dot(h) - dJdq)
        f = Lambda.dot(ddx_des[:,i]) + mu
        
        # secondary task
        J_T_pinv = Lambda.dot(J_Minv)
        NJ = np.eye(robot.nv) - J.T.dot(J_T_pinv)
        ddq_pos_des = kp_j * (conf.q0 - q[:,i]) - kd_j*v[:,i]
        tau_0 = M.dot(ddq_pos_des)


        k = 8

        # # define the control laws here
        if(test['controller']=='IC_O_simpl'):
            tau[:,i] = h + J.T.dot(k*ddx_fb)

        elif(test['controller']=='IC_O_simpl_post'):
            tau[:,i] = h + J.T.dot(k*ddx_fb) + NJ.dot(tau_0)

        elif(test['controller']=='IC_O'):
            tau[:,i] = J.T.dot(ddx_fb + mu)

        elif(test['controller']=='IC_O_post'):                                                                 
            tau[:,i] = J.T.dot(k*ddx_fb + mu) + NJ.dot(tau_0)

        elif(test['controller']=='OSC'):      # Operational Space Control
            tau[:,i] = J.T.dot(f) + NJ.dot(tau_0 + h)

        elif(test['controller']=='IC'):     # Impedence Control
            tau[:,i] = h + J.T.dot(k*ddx_fb) + NJ.dot(tau_0)

        else:
            print('ERROR: Unknown controller', test['controller'])
            sys.exit(0)
        
        # send joint torques to simulator
        simu.simulate(tau[:,i], conf.dt, conf.ndt)
        tau_c[:,i] = simu.tau_c
        ddx[:,i] = J.dot(simu.dv) + dJdq
        t += conf.dt
            
        time_spent = time.time() - time_start
        if(conf.simulate_real_time and time_spent < conf.dt): 
            time.sleep(conf.dt-time_spent)
    
    tracking_err = np.sum(norm(x_ref-x, axis=0))/N

    desc = test['controller']+' kp='+str(test['kp'])+' frequency=[{},{},{}]'.format(test['frequency'][0],test['frequency'][1],test['frequency'][2])

    if conf.TRACK_TRAJ:
        if(test['controller']=='OSC'):        
            tracking_err_osc += [{'value': tracking_err, 'description': desc}]
        elif(test['controller']=='IC'):
            tracking_err_ic += [{'value': tracking_err, 'description': desc}]    
        else:
            print('ERROR: Unknown controller', test['controller'])
    else:
        if(test['controller']=='IC_O_simpl'):
            stab_err_ic_O_simpl += [{'value': tracking_err, 'description': desc}]
        elif(test['controller']=='IC_O_simpl_post'):
            stab_err_ic_O_simpl_post += [{'value': tracking_err, 'description': desc}]
        elif(test['controller']=='IC_O'):
            stab_err_ic_O += [{'value': tracking_err, 'description': desc}]
        elif(test['controller']=='IC_O_post'):
            stab_err_ic_O_post += [{'value': tracking_err, 'description': desc}]
        else:
            print('ERROR: Unknown controller', test['controller'])
    
    print('Average tracking error %.3f m\n'%(tracking_err))
    
    # PLOT STUFF
    tt = np.arange(0.0, N*conf.dt, conf.dt)
    
    if(PLOT_EE_POS):    
        (f, ax) = plut.create_empty_figure(nx)
        ax = ax.reshape(nx)
        for i in range(nx):
            ax[i].plot(tt, x[i,:], label='x')
            ax[i].plot(tt, x_ref[i,:], '--', label='x ref')
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel(r'x_'+str(i)+' [m]')
        leg = ax[0].legend()
        leg.get_frame().set_alpha(0.5)
        f.suptitle(description,y=1)
        
    if(PLOT_EE_VEL):    
        (f, ax) = plut.create_empty_figure(nx)
        ax = ax.reshape(nx)
        for i in range(nx):
            ax[i].plot(tt, dx[i,:], label='dx')
            ax[i].plot(tt, dx_ref[i,:], '--', label='dx ref')
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel(r'dx_'+str(i)+' [m/s]')
        leg = ax[0].legend()
        leg.get_frame().set_alpha(0.5)
        f.suptitle(description,y=1)
       
    if(PLOT_EE_ACC):    
        (f, ax) = plut.create_empty_figure(nx)
        ax = ax.reshape(nx)
        for i in range(nx):
            ax[i].plot(tt, ddx[i,:], label='ddx')
            ax[i].plot(tt, ddx_ref[i,:], '--', label='ddx ref')
            ax[i].plot(tt, ddx_des[i,:], '-.', label='ddx des')
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel(r'ddx_'+str(i)+' [m/s^2]')
        leg = ax[0].legend()
        leg.get_frame().set_alpha(0.5)
        f.suptitle(description,y=1)
         
    if(PLOT_TORQUES):    
        (f, ax) = plut.create_empty_figure(int(robot.nv/2),2)
        ax = ax.reshape(robot.nv)
        for i in range(robot.nv):
            ax[i].plot(tt, tau[i,:], label=r'$\tau$ '+str(i))
            ax[i].plot(tt, tau_c[i,:], label=r'$\tau_c$ '+str(i))
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel('Torque [Nm]')
        leg = ax[0].legend()
        leg.get_frame().set_alpha(0.5)
        f.suptitle(description,y=1)

(f, ax) = plut.create_empty_figure()
if conf.TRACK_TRAJ:
    for (i,err) in enumerate(tracking_err_osc):
        ax.plot(0, err['value'], 's', markersize=20, label=err['description'])
    for (i,err) in enumerate(tracking_err_ic):
        ax.plot(1, err['value'], 'o', markersize=20, label=err['description'])
else:
    for (i,err) in enumerate(stab_err_ic_O_simpl):
        ax.plot(0, err['value'], 's', markersize=20, label=err['description'])
    for (i,err) in enumerate(stab_err_ic_O_simpl_post):
        ax.plot(1, err['value'], 's', markersize=20, label=err['description'])
    for (i,err) in enumerate(stab_err_ic_O):
        ax.plot(2, err['value'], 's', markersize=20, label=err['description'])
    for (i,err) in enumerate(stab_err_ic_O_post):
        ax.plot(3, err['value'], 's', markersize=20, label=err['description']) 
ax.set_xlabel('Test')
ax.set_ylabel('Mean tracking error [m]')
leg = ax.legend()
leg.get_frame().set_alpha(0.5)

plt.show()
