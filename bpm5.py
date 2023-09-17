import numpy as np
from scipy.ndimage import zoom
import jax
from jax.scipy.signal import convolve
#from brainpy.math import convolve
import time
import matplotlib.pyplot as plt
##

from jax.lax import conv_general_dilated

#init=np.load('init.npz')
### update synapse parameter using Runge-Kutta
def RK2_simple_linear_eq(y_vec,delta_t,deriv_coeff,delta_fun_vec):
    # Integration step using Runga-Kutta order 2 method
    y_vec=bm.dot(y_vec,(1+delta_t*deriv_coeff+delta_t**2*deriv_coeff**2/2))+delta_fun_vec
    return y_vec
    
### OU noise for neurons
def OU(I_noise,num_neuron):
    tao_AMPA=2
    thw_noise=4    
    H=5
    gaosi_noise = bm.random.randn(1,num_neuron)
    A = ((1 - bm.exp(-2*H/tao_AMPA))*thw_noise**2/2)**0.5
    I_noise = I_noise*bm.exp(-H/tao_AMPA) + A*gaosi_noise 
    return I_noise
    
# Temporal kernel equation 
def K(t,tau0,tau1):
    kk= t**6 /tau0**7 * bm.exp( -t/tau0 ) - t**6 /tau1**7 * bm.exp( -t/tau1 )
    return kk
    
### generate temporal kernel
def get_tk_time_series( ts, t_delay, ONOFF_type, kernel_ab ):
    # ab notation
    a = kernel_ab[0]
    b = kernel_ab[1]

    # Temporal kernel parameters
    tau0=3.66
    tau1=7.16
    #tau0=0.366*0.5
    #tau1=0.716*0.5
    
    
    if ONOFF_type=='ON':
        ks = K(ts-t_delay,tau0,tau1)
        pos_part = ks>0
        pos_part =bm.array(pos_part,dtype=bm.int32)
        ks[pos_part] = a * ks[pos_part]
        neg_part = ks<0
        neg_part =bm.array(neg_part,dtype=bm.int32)
        ks[neg_part] = b * ks[neg_part]
    else:
        ks = -K(ts-t_delay,tau0,tau1)
        neg_part = ks>0
        neg_part =bm.array(neg_part,dtype=bm.int32)
        ks[neg_part] = a * ks[neg_part]
        pos_part = ks<0
        pos_part =bm.array(pos_part,dtype=bm.int32)
        ks[pos_part] = b * ks[pos_part]
    a=ts < t_delay
    a=bm.array(a,dtype=bm.int32)
    ks[a] = 0
    
    return ks

# Spatial kernel equation
def SK (x,y,alpha,beta,sigmaa,sigmab):
    aa=alpha/(bm.pi*sigmaa**2) * bm.exp(-(x**2+y**2)/sigmaa**2) -beta /(bm.pi*sigmab**2) * bm.exp(-(x**2+y**2)/sigmab**2)
    return aa
### generate spatial kernel
def get_SF_factor(scale):

    # Spatial kernel parameters
    alpha=1.0
    beta=0.74
    sigmaa=0.0894
    sigmab=0.1259
    #place
    dx = 0.01
    #xs = -.04:dx:.05
    xs=bm.linspace(-0.05,0.05,5)
    ys = xs
    [Xs,Ys] = bm.meshgrid(xs,ys)
    # Store SF kernel in 2d
    As = SK(Xs,Ys,alpha,beta,sigmaa,sigmab)
    return As

def RK2_4sNMDA(y_vec,delta_t,C_coeff,D_vec):
# Integration step using Runga-Kutta order 2 method,
    y_vec_temp=y_vec+0.5*delta_t*(C_coeff*y_vec+D_vec*(1-y_vec))
    y_vec=y_vec+delta_t*(C_coeff*y_vec_temp+D_vec*(1-y_vec_temp))
    return y_vec

# 从当前刺激中计算出II1(I_on)和II2(I_off)-->LGN input
# get LGN input
def realtime_stim(Ls,stim):
    xy_on=np.load('xy_white_points.npy').T  #200个卷积核的中心位置坐标
    xy_off=np.load('xy_black_points.npy').T
    xy_on=bm.array(xy_on,dtype=bm.int32)
    xy_off=bm.array(xy_off,dtype=bm.int32)
    #the calculation reflect effect of previous 200ms on LGN
    Ls[:,:,0:199]=Ls[:,:,1:200]
    Ls[:,:,199]=stim
    # NN_on=bm.zeros((200,1))
    # NN_off=bm.zeros((200,1))
    NN_on=bm.Variable((200,1))
    NN_off=bm.Variable((200,1))
    ts_kern=bm.linspace(1,201,200)


    ## build ON grid
    t_delay=60
    ONOFF_type='ON'
    #kernel_ab=[1.6 0.7]
    kernel_ab=[1.6,0.7]

    ks=get_tk_time_series(ts_kern, t_delay, ONOFF_type, kernel_ab )
    aas = get_SF_factor(5)
    #scale=int(Ls.shape[0]/aas.shape[0])
    #ass=np.tile(aas,(scale_lgn,scale_lgn))
    #aass=np.dot(ass.reshape(Ls.shape[0],Ls.shape[0],1),np.ones((1,Ls.shape[2])))
    #Lss=aass*Ls
    #Ls_on=bm.zeros((Ls.shape[0],Ls.shape[1],200))
    Ls_on=convolve(Ls.value,ks.reshape(1,1,ks.size).value,'valid') ##convolute spatial temporal kernel
    #Ls1=np.array(np.hsplit(np.array(np.vsplit(Ls_on,scale_lgn)).transpose(1,2,3,0),scale_lgn)).transpose(1,2,4,0,3)
    Ls_on=bm.array(Ls_on)
    ## build OFF grid
    t_delay_off=0
    ONOFF_type_off='OFF'
    kernel_ab_off=[1,1]
    ks_off=get_tk_time_series( ts_kern, t_delay_off, ONOFF_type_off, kernel_ab_off)
#     Ls_off=bm.zeros((Ls.shape[0],Ls.shape[1],200))
#     for i in range (0,Ls.shape[0]):
#         for j in range (0,Ls.shape[1]):
    Ls_off=convolve(Ls.value, ks_off.reshape(1,1,ks_off.size).value,'valid') ##convolute spatial temporal kernel
    #Ls2=np.array(np.hsplit(np.array(np.vsplit(Ls_off,scale_lgn)).transpose(1,2,3,0),scale_lgn)).transpose(1,2,4,0,3)
    Ls_off=bm.array(Ls_off)
    ### calculate input current to LGN
    #NN_on=np.sum(np.sum(Ls1,axis=0),axis=0)
    #NN_off=np.sum(np.sum(Ls2,axis=0),axis=0)

#         print(Ls1.shape)
#         print(NN_on.shape)

    # 循环把卷积核中的数值sum一下
    for i in range (0,200):
        x_on=xy_on[i,0]
        y_on=xy_on[i,1]
        x_off=xy_on[i,0]
        y_off=xy_on[i,1]
        #print(x_on,y_on)
        NN_on[i]=bm.sum(aas*jax.lax.dynamic_slice(Ls_on.value,(5+x_on-3,5+y_on-3,0),(5,5,1)).reshape(5,5))
        NN_off[i]=bm.sum(aas*jax.lax.dynamic_slice(Ls_off.value,(5+x_off-3,5+y_off-3,0),(5,5,1)).reshape(5,5))

    II1=NN_on
    a = II1>0
    a =bm.array(a,dtype=bm.int32)
    II1[a]=0
    II1=0.5*bm.log(II1+1)

    II2=NN_off
    a = II2>0
    a =bm.array(a,dtype=bm.int32)
    II2[a]=0
    II2=0.5*bm.log(II2+1)
    net.tt=net.tt+1
    return Ls,II1, II2
     
# 获取I_on和I_off
# 对输入的变量进行设置，设置为brainpy能够读取的格式   
def set_input(tdi,Ls,s,tt):
    #t = tdi['t']
    #net.tt+=t
    Ls,II1,II2=realtime_stim(Ls,s[:,:,bm.floor(tt/10).astype('int32')].reshape(110,110))
    II1=II1.reshape(200,)*0.25
    II2=II2.reshape(200,)*0.25
    #print(np.mean(II1))

    #net.lgn_pop_on.input.value = II1.astype(bm.float32)
    #net.lgn_pop_off.input.value = II2.astype(bm.float32)
    net.lgn_pop_on.input = II1.astype(bm.float32)
    net.lgn_pop_off.input = II2.astype(bm.float32)
    net.Ls=Ls  
    net.II1=II1
    net.II2=II2
    net.tt+=1
    # print(type(II1))
    # print(II1.value)
    
    
import brainpy as bp
import brainpy.math as bm
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from brainpy import synapses
from brainpy.connect import SparseMatConn
class LGN_V1_MT_LIP(bp.Network):
    def __init__(self, scale=1., mu0=40., coherence=25.6, f=0.15):
        super(LGN_V1_MT_LIP, self).__init__()
        # Define the parameters for the population
        
        scale=20
        num_lgn_on = 200#1600
        num_lgn_off =200
        num_v1_e = 90000#int(281548/20)
        num_v1_i = 22500#int(70388/20)
        num_mt_e = 17956#int(56404/20)
        num_mt_i = 4489#int(14100/20)
        num_lip_e = 8464
        num_lip_i = 2116
        
        ampa_par = dict(delay_step=5,tau=2.0)
        gaba_par = dict(delay_step=5,tau=5.0)
        nmda_par = dict(delay_step=5,tau_decay=100, tau_rise=2., a=0.5)
        
        # Define the differential equation for LIF neurons
        lif_params = {
            'V_rest': -70.,   # rest voltage 静息电压
            'V_reset': -60.,   # reset voltage 重置电压
            'V_th': -50.,  # threshold voltage
            'tau_m': 20.,  # time constant
            't_refractory': 2.  # refractory period
        }
        self.Ls=bm.Variable((110,110,200))
        self.II1=bm.Variable((200,))
        self.II2=bm.Variable((200,))
        self.tt=bm.Variable(1, dtype='int32')
        self.noise_lgn_on=bp.neurons.PoissonGroup(num_lgn_on, freqs=0)
        self.noise_lgn_off=bp.neurons.PoissonGroup(num_lgn_off, freqs=0)
        self.noise_v1_e=bp.neurons.PoissonGroup(num_v1_e, freqs=0)
        self.noise_v1_i=bp.neurons.PoissonGroup(num_v1_i, freqs=0)
        self.noise_mt_e=bp.neurons.PoissonGroup(num_mt_e, freqs=0)
        self.noise_mt_i=bp.neurons.PoissonGroup(num_mt_i, freqs=0)
        self.noise_lip_e=bp.neurons.PoissonGroup(num_lip_e, freqs=0)
        self.noise_lip_i=bp.neurons.PoissonGroup(num_lip_i, freqs=0)
        
        # 神经元初始化
        self.lgn_pop_on = bp.neurons.LIF(size=num_lgn_on,tau=lif_params['tau_m'], V_rest=lif_params['V_rest'], V_reset=lif_params['V_reset'], V_th=lif_params['V_th'],tau_ref=2.,R=0.05,V_initializer=bp.init.Normal(-53,0.7)) 
#                              tau_ref=lif_params['t_refractory'])
#         self.lgn_pop_on = bp.neurons.LIF(tau=lif_params['tau_m'], V_reset=lif_params['V_reset'], V_th=lif_params['V_th'], 
#                              tau_ref=lif_params['t_refractory'],size=num_lgn_on)
        self.lgn_pop_off = bp.neurons.LIF(tau=lif_params['tau_m'], V_rest=lif_params['V_rest'], V_reset=lif_params['V_reset'], V_th=lif_params['V_th'],size=num_lgn_off,tau_ref=2.,R=0.05,V_initializer=bp.init.Normal(-53,0.7)) 
        self.v1_pop_e = bp.neurons.LIF(tau=lif_params['tau_m'], V_rest=lif_params['V_rest'], V_reset=lif_params['V_reset'], V_th=lif_params['V_th'],size=num_v1_e,tau_ref=2.,R=0.05,V_initializer=bp.init.Normal(-53,0.7)) 
        self.v1_pop_i = bp.neurons.LIF(tau=lif_params['tau_m'], V_rest=lif_params['V_rest'], V_reset=lif_params['V_reset'], V_th=lif_params['V_th'],size=num_v1_i,tau_ref=2.,R=0.05,V_initializer=bp.init.Normal(-53,0.7)) 
        self.mt_pop_e = bp.neurons.LIF(tau=lif_params['tau_m'], V_rest=lif_params['V_rest'], V_reset=lif_params['V_reset'], V_th=lif_params['V_th'],size=num_mt_e,tau_ref=2.,R=0.05,V_initializer=bp.init.Normal(-53,0.7)) 
        self.mt_pop_i = bp.neurons.LIF(tau=lif_params['tau_m'], V_rest=lif_params['V_rest'], V_reset=lif_params['V_reset'], V_th=lif_params['V_th'],size=num_mt_i,tau_ref=2.,R=0.05,V_initializer=bp.init.Normal(-53,0.7)) 
        self.lip_pop_e = bp.neurons.LIF(tau=lif_params['tau_m'], V_rest=lif_params['V_rest'], V_reset=lif_params['V_reset'], V_th=lif_params['V_th'],size=num_lip_e,tau_ref=2.,R=0.05,V_initializer=bp.init.Normal(-53,0.7)) 
        self.lip_pop_i = bp.neurons.LIF(tau=lif_params['tau_m'], V_rest=lif_params['V_rest'], V_reset=lif_params['V_reset'], V_th=lif_params['V_th'],size=num_lip_i,tau_ref=2.,R=0.05,V_initializer=bp.init.Normal(-53,0.7)) 

#         self.Ion = PoissonStim(num_lgn_on, freq_var=2, t_interval=2., freq_mean=II1)
#         self.Ioff = PoissonStim(num_lgn_off, freq_var=2, t_interval=2., freq_mean=II2)

        # Define weight matrix for intra-V1 connections
        # 定义每一层神经元之间连接的权重矩阵
        lgnon_v1_weight_matrix=np.load('Weight_lgn_v1_on_sparse.npz')
        scale_onv1=bm.sqrt(lgnon_v1_weight_matrix['data'].shape[0]/num_v1_e)
        lgnon_v1_weight_matrix=csr_matrix((lgnon_v1_weight_matrix['data'],lgnon_v1_weight_matrix['indices'],lgnon_v1_weight_matrix['indptr']),shape=tuple(lgnon_v1_weight_matrix['shape']))
        lgnoff_v1_weight_matrix=np.load('Weight_lgn_v1_off_sparse.npz')
        scale_offv1=lgnoff_v1_weight_matrix['data'].shape[0]/num_v1_e
        lgnoff_v1_weight_matrix=csr_matrix((lgnoff_v1_weight_matrix['data'],lgnoff_v1_weight_matrix['indices'],lgnoff_v1_weight_matrix['indptr']),shape=tuple(lgnoff_v1_weight_matrix['shape']))
        v1v1_weight_matrix=np.load('weight_v1v1_sparse.npz')
        scale_v1v1=bm.sqrt(v1v1_weight_matrix['data'].shape[0]/num_v1_e)
        v1v1_weight_matrix=csr_matrix((v1v1_weight_matrix['data'],v1v1_weight_matrix['indices'],v1v1_weight_matrix['indptr']),shape=tuple(v1v1_weight_matrix['shape']))
        v1v1I_weight_matrix=np.load('weight_v1v1I_sparse.npz')
        scale_v1v1I=bm.sqrt(v1v1I_weight_matrix['data'].shape[0]/num_v1_i)
        v1v1I_weight_matrix=csr_matrix((v1v1I_weight_matrix['data'],v1v1I_weight_matrix['indices'],v1v1I_weight_matrix['indptr']),shape=tuple(v1v1I_weight_matrix['shape']))
        v1mt_weight_matrix=np.load('weight_v1mt_sparse.npz')
        scale_v1mt=bm.sqrt(v1mt_weight_matrix['data'].shape[0]/num_mt_e)
        v1mt_weight_matrix=csr_matrix((v1mt_weight_matrix['data'],v1mt_weight_matrix['indices'],v1mt_weight_matrix['indptr']),shape=tuple(v1mt_weight_matrix['shape']))
        mtmt_weight_matrix=np.load('weight_mtmt_sparse.npz')
        scale_mtmt=bm.sqrt(mtmt_weight_matrix['data'].shape[0]/num_mt_e)
        mtmt_weight_matrix=csr_matrix((mtmt_weight_matrix['data'],mtmt_weight_matrix['indices'],mtmt_weight_matrix['indptr']),shape=tuple(mtmt_weight_matrix['shape']))
        mtmtI_weight_matrix=np.load('weight_mtmtI_sparse.npz')
        scale_mtmtI=bm.sqrt(mtmtI_weight_matrix['data'].shape[0]/num_mt_i)
        mtmtI_weight_matrix=csr_matrix((mtmtI_weight_matrix['data'],mtmtI_weight_matrix['indices'],mtmtI_weight_matrix['indptr']),shape=tuple(mtmtI_weight_matrix['shape']))
        mtlip_weight_matrix=np.load('weight_mtlip_sparse.npz')
        scale_mtlip=bm.sqrt(mtlip_weight_matrix['data'].shape[0]/num_lip_e)
        mtlip_weight_matrix=csr_matrix((mtlip_weight_matrix['data'],mtlip_weight_matrix['indices'],mtlip_weight_matrix['indptr']),shape=tuple(mtlip_weight_matrix['shape']))
        liplip_weight_matrix=np.load('weight_liplip_sparse.npz')
        scale_liplip=bm.sqrt(liplip_weight_matrix['data'].shape[0]/num_lip_e)
        liplip_weight_matrix=csr_matrix((liplip_weight_matrix['data'],liplip_weight_matrix['indices'],liplip_weight_matrix['indptr']),shape=tuple(liplip_weight_matrix['shape']))
        liplipI_weight_matrix=np.load('weight_liplipI_sparse.npz')
        scale_liplipI=bm.sqrt(liplipI_weight_matrix['data'].shape[0]/num_lip_i)
        liplipI_weight_matrix=csr_matrix((liplipI_weight_matrix['data'],liplipI_weight_matrix['indices'],liplipI_weight_matrix['indptr']),shape=tuple(liplipI_weight_matrix['shape']))
        v1Iv1_weight_matrix=np.load('weight_v1Iv1_sparse.npz')
        scale_v1Iv1=bm.sqrt(v1Iv1_weight_matrix['data'].shape[0]/num_v1_e)
        v1Iv1_weight_matrix=csr_matrix((v1Iv1_weight_matrix['data'],v1Iv1_weight_matrix['indices'],v1Iv1_weight_matrix['indptr']),shape=tuple(v1Iv1_weight_matrix['shape']))
        mtImt_weight_matrix=np.load('weight_mtImt_sparse.npz')
        scale_mtImt=bm.sqrt(mtImt_weight_matrix['data'].shape[0]/num_mt_e)
        mtImt_weight_matrix=csr_matrix((mtImt_weight_matrix['data'],mtImt_weight_matrix['indices'],mtImt_weight_matrix['indptr']),shape=tuple(mtImt_weight_matrix['shape']))
        lipIlip_weight_matrix=np.load('weight_lipIlip_sparse.npz')
        scale_lipIlip=bm.sqrt(lipIlip_weight_matrix['data'].shape[0]/num_lip_e)
        lipIlip_weight_matrix=csr_matrix((lipIlip_weight_matrix['data'],lipIlip_weight_matrix['indices'],lipIlip_weight_matrix['indptr']),shape=tuple(lipIlip_weight_matrix['shape']))
        v1Iv1I_weight_matrix=np.load('weight_v1Iv1I_sparse.npz')
        scale_v1Iv1I=bm.sqrt(v1Iv1I_weight_matrix['data'].shape[0]/num_v1_i)
        v1Iv1I_weight_matrix=csr_matrix((v1Iv1I_weight_matrix['data'],v1Iv1I_weight_matrix['indices'],v1Iv1I_weight_matrix['indptr']),shape=tuple(v1Iv1I_weight_matrix['shape']))
        mtImtI_weight_matrix=np.load('weight_mtImtI_sparse.npz')
        scale_mtImtI=bm.sqrt(mtImtI_weight_matrix['data'].shape[0]/num_mt_i)
        mtImtI_weight_matrix=csr_matrix((mtImtI_weight_matrix['data'],mtImtI_weight_matrix['indices'],mtImtI_weight_matrix['indptr']),shape=tuple(mtImtI_weight_matrix['shape']))
        lipIlipI_weight_matrix=np.load('weight_lipIlipI_sparse.npz')
        scale_lipIlipI=bm.sqrt(lipIlipI_weight_matrix['data'].shape[0]/num_lip_i)
        lipIlipI_weight_matrix=csr_matrix((lipIlipI_weight_matrix['data'],lipIlipI_weight_matrix['indices'],lipIlipI_weight_matrix['indptr']),shape=tuple(lipIlipI_weight_matrix['shape']))

        #         v1_weight_matrix[:num_v1_e, :num_v1_e] = 0.5  # excitatory to excitatory
#         v1_weight_matrix[num_v1_e:, :num_v1_e] = -0.5  # inhibitory to excitatory
#         v1_weight_matrix[:num_v1_e, num_v1_e:] = 1.5  # excitatory to inhibitory
#         v1_weight_matrix[num_v1_e:, num_v1_e:] = -1.5  # inhibitory to inhibitory

#         # Define weight matrix for inter-area connections
#         lgn_v1_weight_matrix = np.random.normal(0.5, 0.1, (num_lgn, num_v1_e))
#         v1_mt_weight_matrix = np.random.normal(0.5, 0.1, (num_v1_e, num_mt_e))
#         mt_lip_weight_matrix = np.random.normal(0.5, 0.1, (num_mt_e, num_lip_e))
#         lip_v1_weight_matrix = np.random.normal(0.5, 0.1, (num_lip_e, num_v1_e))

#         lgn_v1_weight_matrix=csr_matrix(lgn_v1_weight_matrix)
#         v1_mt_weight_matrix=csr_matrix(v1_mt_weight_matrix)
#         mt_lip_weight_matrix=csr_matrix(mt_lip_weight_matrix)
#         lip_v1_weight_matrix=csr_matrix(lip_v1_weight_matrix)
#         self.Ion2on = bp.synapses.Exponential(self.Ion, self.lgn_pop_on, bp.conn.One2One(), g_max=2.1,
#                                             output=bp.synouts.COBA(E=0.))
#         self.Ion2on = bp.synapses.Exponential(self.Ioff, self.lgn_pop_off, bp.conn.One2One(), g_max=2.1,
#                                             output=bp.synouts.COBA(E=0.))
        
        # 初始化突触：噪声&信号
        self.noise_lgn_on_syn=bp.synapses.Exponential(self.noise_lgn_on, self.lgn_pop_on, bp.conn.One2One(), g_max=100,output=bp.synouts.COBA(E=0.), **ampa_par)
        self.noise_lgn_off_syn=bp.synapses.Exponential(self.noise_lgn_off, self.lgn_pop_off, bp.conn.One2One(), g_max=100,output=bp.synouts.COBA(E=0.), **ampa_par)
        self.noise_v1_e_syn=bp.synapses.Exponential(self.noise_v1_e, self.v1_pop_e, bp.conn.One2One(), g_max=100,output=bp.synouts.COBA(E=0.), **ampa_par)
        self.noise_v1_i_syn=bp.synapses.Exponential(self.noise_v1_i, self.v1_pop_i, bp.conn.One2One(), g_max=100,output=bp.synouts.COBA(E=0.), **ampa_par)
        self.noise_mt_e_syn=bp.synapses.Exponential(self.noise_mt_e, self.mt_pop_e, bp.conn.One2One(), g_max=100,output=bp.synouts.COBA(E=0.), **ampa_par)
        self.noise_mt_i_syn=bp.synapses.Exponential(self.noise_mt_i, self.mt_pop_i, bp.conn.One2One(), g_max=100,output=bp.synouts.COBA(E=0.), **ampa_par)
        self.noise_lip_e_syn=bp.synapses.Exponential(self.noise_lip_e, self.lip_pop_e, bp.conn.One2One(), g_max=100,output=bp.synouts.COBA(E=0.), **ampa_par)
        self.noise_lip_i_syn=bp.synapses.Exponential(self.noise_lip_i, self.lip_pop_i, bp.conn.One2One(), g_max=100,output=bp.synouts.COBA(E=0.), **ampa_par)
        sss=0.56#0.2#0.7
        sss1=0.35*0.5
        sss2=0.35
        
        # 神经元之间的突触连接初始化，包括了自连接以及兴奋or抑制
        self.lgnon_v1_synapses = synapses.Exponential(self.lgn_pop_on, self.v1_pop_e, SparseMatConn(lgnon_v1_weight_matrix),g_max=1.5*4000/scale_onv1*sss,output=bp.synouts.COBA(E=0.), **ampa_par)
        self.lgnoff_v1_synapses = synapses.Exponential(self.lgn_pop_off, self.v1_pop_e, SparseMatConn(lgnoff_v1_weight_matrix),g_max=1.5*4000/scale_offv1*sss,output=bp.synouts.COBA(E=0.), **ampa_par)
        self.v1v1_synapses = synapses.Exponential(self.v1_pop_e, self.v1_pop_e, SparseMatConn(v1v1_weight_matrix),g_max=1.2*700/scale_v1v1*sss,output=bp.synouts.COBA(E=0.), **ampa_par)
        self.v1v1I_synapses = synapses.Exponential(self.v1_pop_e, self.v1_pop_i, SparseMatConn(v1v1I_weight_matrix),g_max=200/scale_v1v1I*sss,output=bp.synouts.COBA(E=0.), **ampa_par)
        self.v1v1_NMDA_synapses = synapses.NMDA(self.v1_pop_e, self.v1_pop_e, conn=SparseMatConn(v1v1_weight_matrix),g_max=1.2*300/scale_v1v1*sss,output=bp.synouts.MgBlock(E=0., cc_Mg=1.),**nmda_par)
        self.v1v1I_NMDA_synapses = synapses.NMDA(self.v1_pop_e, self.v1_pop_i, conn=SparseMatConn(v1v1I_weight_matrix),g_max=100/scale_v1v1I**sss,output=bp.synouts.MgBlock(E=0., cc_Mg=1.),**nmda_par)
        self.v1Iv1_synapses = synapses.Exponential(self.v1_pop_i, self.v1_pop_e, conn=SparseMatConn(v1Iv1_weight_matrix),g_max=0.4*500/scale_v1Iv1**sss,output=bp.synouts.COBA(E=-70.),**gaba_par)
        self.v1Iv1I_synapses = synapses.Exponential(self.v1_pop_i, self.v1_pop_i, conn=SparseMatConn(v1Iv1I_weight_matrix),g_max=0.4*500/scale_v1Iv1I*sss,output=bp.synouts.COBA(E=-70.),**gaba_par)
        self.v1mt_synapses = synapses.Exponential(self.v1_pop_e, self.mt_pop_e, SparseMatConn(v1mt_weight_matrix),g_max=900/scale_v1mt*sss1,output=bp.synouts.COBA(E=0.), **ampa_par)
        self.mtmt_synapses = synapses.Exponential(self.mt_pop_e, self.mt_pop_e, conn=SparseMatConn(mtmt_weight_matrix),g_max=1000/scale_mtmt*sss1,output=bp.synouts.COBA(E=0.), **ampa_par)
        self.mtmtI_synapses = synapses.Exponential(self.mt_pop_e, self.mt_pop_i, conn=SparseMatConn(mtmtI_weight_matrix),g_max=500/scale_mtmtI*sss1,output=bp.synouts.COBA(E=0.), **ampa_par)
        self.mtmt_NMDA_synapses = synapses.NMDA(self.mt_pop_e, self.mt_pop_e, conn=SparseMatConn(mtmt_weight_matrix),g_max=9000/scale_mtmt*sss1,output=bp.synouts.MgBlock(E=0., cc_Mg=1.),**nmda_par)
        self.mtmtI_NMDA_synapses = synapses.NMDA(self.mt_pop_e, self.mt_pop_i, conn=SparseMatConn(mtmtI_weight_matrix),g_max=9000/scale_mtmtI*sss1,output=bp.synouts.MgBlock(E=0., cc_Mg=1.),**nmda_par)
        self.mtImt_synapses = synapses.Exponential(self.mt_pop_i, self.mt_pop_e, conn=SparseMatConn(mtImt_weight_matrix),g_max=7000/scale_mtImt*sss1,output=bp.synouts.COBA(E=-70.),**gaba_par)
        self.mtImtI_synapses = synapses.Exponential(self.mt_pop_i, self.mt_pop_i, conn=SparseMatConn(mtImtI_weight_matrix),g_max=7000/scale_mtImtI*sss1,output=bp.synouts.COBA(E=-70.),**gaba_par)
        self.mtlip_synapses = synapses.Exponential(self.mt_pop_e, self.lip_pop_e, SparseMatConn(mtlip_weight_matrix),g_max=300/scale_mtlip*sss1,output=bp.synouts.COBA(E=0.), **ampa_par)
        self.liplip_synapses = synapses.Exponential(self.lip_pop_e, self.lip_pop_e, conn=SparseMatConn(liplip_weight_matrix),g_max=4000/scale_liplip*sss2,output=bp.synouts.COBA(E=0.), **ampa_par)
        self.liplipI_synapses = synapses.Exponential(self.lip_pop_e, self.lip_pop_i, conn=SparseMatConn(liplipI_weight_matrix),g_max=11000/scale_liplipI*sss2,output=bp.synouts.COBA(E=0.), **ampa_par)
        self.liplip_NMDA_synapses = synapses.NMDA(self.lip_pop_e, self.lip_pop_e, conn=SparseMatConn(liplip_weight_matrix),g_max=7000/scale_liplip*sss2,output=bp.synouts.MgBlock(E=0., cc_Mg=1.),**nmda_par)
        self.liplipI_NMDA_synapses = synapses.NMDA(self.lip_pop_e, self.lip_pop_i, conn=SparseMatConn(liplipI_weight_matrix),g_max=7000/scale_liplipI*sss2,output=bp.synouts.MgBlock(E=0., cc_Mg=1.),**nmda_par)
        self.lipIlip_synapses = synapses.Exponential(self.lip_pop_i, self.lip_pop_e, conn=SparseMatConn(lipIlip_weight_matrix),g_max=8000/scale_lipIlip*sss2,output=bp.synouts.COBA(E=-70.),**gaba_par)
        self.lipIlipI_synapses = synapses.Exponential(self.lip_pop_i, self.lip_pop_i, conn=SparseMatConn(lipIlipI_weight_matrix),g_max=8000/scale_lipIlipI*sss2,output=bp.synouts.COBA(E=-70.),**gaba_par)
import os
import brainpy.math as bm
import numpy as np
#bm.dt=1
# II1=np.load('I_on.npy').T
# II2=np.load('I_off.npy').T
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
net = LGN_V1_MT_LIP()

# 先定义一个20帧的0矩阵
s=np.concatenate((np.zeros((110,110,20)),np.load('stimulus_90.npy')),2)
#s=np.zeros((110,110,120))
s=bm.array(s)
runner = bp.DSRunner(net,
                   monitors=['noise_lgn_on.spike','lgn_pop_on.spike','lgn_pop_off.spike','v1_pop_e.spike', 'lgn_pop_on.V','mt_pop_e.spike', 'lip_pop_e.spike','tt','Ls','II1','II2','lgn_pop_on.input'],
                   inputs=lambda tdi: set_input(tdi,net.Ls,s,net.tt),dt=0.1,    # functional inputs
                   jit=True)
# t=60,dt=0.1,所以有600 steps
runner.run(60)
np.save('spike.npy',runner.mon['v1_pop_e.spike'])

bp.visualize.raster_plot(runner.mon.ts,runner.mon['noise_lgn_on.spike'])
plt.savefig('lgn.png')
np.save('v1spike.npy',runner.mon['v1_pop_e.spike'])
np.save('mtspike.npy',runner.mon['mt_pop_e.spike'])
np.save('lipspike.npy',runner.mon['lip_pop_e.spike'])
bp.visualize.raster_plot(runner.mon.ts,runner.mon['v1_pop_e.spike'][:,0:200])
plt.savefig('v1.png')
bp.visualize.raster_plot(runner.mon.ts,runner.mon['mt_pop_e.spike'][:,0:200])
plt.savefig('mt.png')
bp.visualize.raster_plot(runner.mon.ts,runner.mon['lip_pop_e.spike'][:,0:200])
plt.savefig('lip.png')

num_lgn_on = 200#1600
num_lgn_off =200
num_v1_e = 90000#int(281548/20)
num_v1_i = 22500#int(70388/20)
num_mt_e = 17956#int(56404/20)
num_mt_i = 4489#int(14100/20)
num_lip_e = 8464
num_lip_i = 2116
plt.figure(figsize=(5,5))
rate_lgn=bp.measure.firing_rate(runner.mon['lgn_pop_on.spike'],width=10)
rate_v1=bp.measure.firing_rate(runner.mon['v1_pop_e.spike'],width=10)
rate_mt=bp.measure.firing_rate(runner.mon['mt_pop_e.spike'],width=10)
rate_lip=bp.measure.firing_rate(runner.mon['lip_pop_e.spike'],width=10)
plt.plot(rate_lgn,label='lgn')
plt.plot(rate_v1,label='v1')
plt.plot(rate_mt,label='mt')
plt.plot(rate_lip,label='lip')
plt.xlabel('time',fontsize=20)
plt.ylabel('fr (Hz)',fontsize=20)
plt.legend()
plt.savefig('fr.png')

plt.figure(figsize=(5,5))
plt.plot(runner.mon['lgn_pop_on.V'][:,77])
plt.plot(runner.mon['lgn_pop_on.input'][:,77])
plt.savefig('v1vm.png')

plt.figure(figsize=(5,5))
plt.imshow(runner.mon['Ls'][400,:,:,199].reshape(110,110))
plt.savefig('Ls.png')
#np.save('Ls.npy',runner.mon['Ls'])


plt.figure(figsize=(5,5))
angle=np.load('v1_angle.npy')
a=np.where(angle==0)[0]
b=np.where(angle==1)[0]
c=np.where(angle==2)[0]
d=np.where(angle==3)[0]
rate_1=bp.measure.firing_rate(runner.mon['v1_pop_e.spike'][:,a],width=10)
rate_2=bp.measure.firing_rate(runner.mon['v1_pop_e.spike'][:,b],width=10)
rate_3=bp.measure.firing_rate(runner.mon['v1_pop_e.spike'][:,c],width=10)
rate_4=bp.measure.firing_rate(runner.mon['v1_pop_e.spike'][:,d],width=10)
plt.plot(rate_1,label='270')
plt.plot(rate_2,label='90')
plt.plot(rate_3,label='0')
plt.plot(rate_4,label='180')
plt.xlabel('time',fontsize=20)
plt.ylabel('fr (Hz)',fontsize=20)
plt.legend()
plt.savefig('angle.png')


plt.figure(figsize=(5,5))
plt.imshow(np.sum(runner.mon['v1_pop_e.spike'],axis=0).reshape(300,300),cmap='YlOrRd')
plt.colorbar()
plt.savefig('v1h.png')
np.save('v1h.npy',np.sum(runner.mon['v1_pop_e.spike'],axis=0).reshape(300,300))
plt.figure(figsize=(5,5))
plt.imshow(np.sum(runner.mon['mt_pop_e.spike'],axis=0).reshape(134,134),cmap='YlOrRd')
plt.colorbar()
plt.savefig('mth.png')
np.save('mth.npy',np.sum(runner.mon['mt_pop_e.spike'],axis=0).reshape(134,134))
plt.figure(figsize=(5,5))
plt.imshow(np.sum(runner.mon['lip_pop_e.spike'],axis=0).reshape(92,92),cmap='YlOrRd')
plt.colorbar()
plt.savefig('liph.png')
np.save('liph.npy',np.sum(runner.mon['lip_pop_e.spike'],axis=0).reshape(92,92))


