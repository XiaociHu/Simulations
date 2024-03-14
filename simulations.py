#!/bin/python3

import os
import numpy as np
from multiprocessing import Process
import random
from scipy.signal import correlate

#import matplotlib.pylab as plt


class Simulations():

    def __init__(self, N_simulations, Frequencies,  dirname, mass=1, L=5, n_atoms=1024, timestep=0.001, time=25):
      self.N_simulations = N_simulations
      self.Frequencies = Frequencies
      self.frequency = self.Frequencies[0]
      self.mass = mass
      self.L = L
      self.n_atoms = n_atoms
      self.timestep = timestep
      self.time = time
      self.dirname = dirname


    def write_input_config_file(self, dirname = None):
        if dirname is None:
            dirname = self.dirname

        frequency = self.frequency
        run_dump_config = self.N_simulations * 2000
        #print(frequency)
        #print(run_dump_config)
        # file content
        file_content = f"""
variable	temp equal 1.0
variable	rc equal 2.5
variable	freq  equal {frequency:.1f}
units		lj
atom_style 	atomic
dimension 	3

atom_modify map yes
read_data	data.configs                  
variable	force_x equal 0
variable    force_y equal 0
variable    force_z equal "5*sin(2*PI*v_freq*{self.timestep}*step)"

change_box 	all boundary p p p
pair_style 	 lj/cut ${{rc}}
pair_coeff 	 * * 1 1
pair_modify shift yes

velocity	all create ${{temp}} 97287

group 	 particle2 type 2
fix      2 particle2 addforce v_force_x v_force_y v_force_z

unfix    2

min_style	sd
minimize	1.0e-4 1.0e-6 1000 10000
fix		1 all nve
fix		4 all langevin ${{temp}} ${{temp}} 0.1 498094

thermo		100
run		5000

reset_timestep 0
compute		1 all msd
fix		3 all ave/time 1 1 1000 c_1[4] file dump.msd

run		5000
dump	1 all custom 2000 dump.config id type x y z vx vy vz
dump_modify	1 sort id
thermo		1000
run    {run_dump_config:d}
        """

        # debug: correctly formatted
        #print(file_content)
        with open(f"{dirname}/in.configs", "w") as f:
            f.write(file_content)


    def write_input_file(self, dirname = None):
        if dirname is None:
            dirname = self.dirname
        frequency = self.frequency # ensure this is a numerical value
        run_dump_p2 = int(self.time / self.timestep)
        # print(frequency)
        #content
        file_content = f"""
variable rc equal 2.5
variable freq  equal {frequency:.1f}
units			lj
atom_style 	atomic
dimension 	3

atom_modify map yes
read_data	data.start_configuration
variable	force_x equal 0
variable    force_y equal 0
variable    force_z equal "5*sin(2*PI*v_freq*{self.timestep}*step)"
change_box 	all boundary p p p

pair_style 	 lj/cut ${{rc}}
pair_coeff 	 * * 1 1
pair_modify shift yes

group 	 particle2 type 2
fix      2 particle2 addforce v_force_x v_force_y v_force_z

fix		1 all nve

thermo		100
log	log.{frequency:.1f}_k
dump	1 particle2 custom 1 dump.p2 id type x y z vx vy vz fx fy fz
dump_modify	1 format float %20.16g
run	    {run_dump_p2:d} 
        """
        # debug: correctly formatted
        #print(file_content)
        with open(f"{dirname}/in.{frequency}_k", "w") as f:
            f.write(file_content)

    def write_data_config_file(self, dirname = None):
        if dirname is None:
            dirname = self.dirname
        with open(f"{dirname}/data.configs", "w") as f:
            f.write('LAMMPS data file from restart file: timestep = 1, procs = 1 \n\n')
            f.write(f'{self.n_atoms} atoms\n\n')
            f.write(f'{2} atom types\n\n')
            f.write(f' {-self.L} {self.L} xlo xhi \n\n')
            f.write(f' {-self.L} {self.L} ylo yhi \n\n')
            f.write(f' {-self.L} {self.L} zlo zhi \n\n')
            f.write(f'Masses \n\n')
            f.write(f'1 1 \n')
            f.write(f'2 {self.mass} \n')
            f.write('\n')
            f.write('Atoms \n\n')
            for i in range(1, self.n_atoms):#+1):
                r = [np.random.uniform(low = -self.L, high = self.L) for j in range(3)]
                f.write(f'{i} {1} {r[0]} {r[1]} {r[2]}\n')
            r = [np.random.uniform(low = -self.L, high = self.L) for j in range(3)]
            f.write(f'{self.n_atoms} {2} {r[0]} {r[1]} {r[2]}\n')

    def write_data_file(self, config, path):
        n_atoms = len(config)
        with open(f"{path}/data.start_configuration", "w") as f:
            f.write('LAMMPS data file from restart file: timestep = 1, procs = 1 \n\n')
            f.write(f'{self.n_atoms} atoms\n\n')
            f.write(f'{2} atom types\n\n')
            f.write(f' {-self.L} {self.L} xlo xhi \n\n')
            f.write(f' {-self.L} {self.L} ylo yhi \n\n')
            f.write(f' {-self.L} {self.L} zlo zhi \n\n')
            f.write(f'Masses \n\n')
            f.write(f'1 1 \n')
            f.write(f'2 {self.mass} \n')
            f.write('\n')
            f.write('Atoms \n\n')

            for i in config:#+1):
                f.write(f'{int(i[0])} {int(i[1])} {float(i[2])} {float(i[3])} {float(i[4])} \n')
            f.write('Velocities \n\n')
            for i in config:#+1):
                f.write(f'{int(i[0])} {float(i[5])} {float(i[6])} {float(i[7])} \n')

    def create_filestructure(self):
        os.makedirs(self.dirname, exist_ok=True)
        os.makedirs(f'{self.dirname}/configurations', exist_ok=True)
        os.makedirs(f'{self.dirname}/analysis', exist_ok=True)
        for freq in self.Frequencies:
            self.frequency = freq
            os.mkdir(f'{self.dirname}/configurations/{freq}_k')
            os.mkdir(f'{self.dirname}/analysis/{freq}_k')
            os.mkdir(f'{self.dirname}/analysis/{freq}_k/vacf')
            os.mkdir(f'{self.dirname}/analysis/{freq}_k/memory')
            #os.mkdir(f'{self.dirname}/analysis/{freq}_k/ff')
            #os.mkdir(f'{self.dirname}/analysis/{freq}_k/term')
            os.mkdir(f'{self.dirname}/configurations/{freq}_k/create_configurations')
            self.write_input_config_file(f'{self.dirname}/configurations/{freq}_k/create_configurations')
            self.write_data_config_file(f'{self.dirname}/configurations/{freq}_k/create_configurations')
            for i in range(0, self.N_simulations):
                os.mkdir(f'{self.dirname}/configurations/{freq}_k/traj{i}')
                if i == 0:
                    self.write_input_file(f'{self.dirname}/configurations/{freq}_k/traj{i}')

    def create_configurations(self):
        for freq in self.Frequencies:
            os.chdir(f'{self.dirname}/configurations/{freq}_k/create_configurations')
            os.system(f'lmp < in.configs -screen none')
            os.chdir('../../../..')
            config_lines = []
            try:
                with open(f'{self.dirname}/configurations/{freq}_k/create_configurations/dump.config', 'r') as f:
                    add = False
                    for line in f:
                        if 'ITEM: ATOMS' in line:
                            config_lines.append([])
                            add = True
                        elif add:
                            config_lines[-1].append(line.split())
                            if line[:len(str(self.n_atoms))] == str(self.n_atoms):
                                add = False
                for i, config in enumerate(config_lines):
                    self.write_data_file(config, f'{self.dirname}/configurations/{freq}_k/traj{i}')
            except FileNotFoundError:
                print("File not found:",os.getcwd())

    def start_simulations(self):
        processes = []
        for freq in self.Frequencies:
            print(freq)
            p = Process(target = self.start_simulation, args = (freq,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    def start_simulation(self, freq):
        os.chdir(f'{self.dirname}/configurations/{freq}_k')
        for i in range(0, self.N_simulations):
            print(i)
            os.chdir(f'traj{i}')
            os.system(f'lmp < in.{freq}_k -screen none')
            self.reformat_data(".")
            os.remove('dump.p2')
            os.chdir('..')
            if i != self.N_simulations-1:
                os.rename(f'traj{i}/in.{freq}_k', f"traj{i+1}/in.{freq}_k")
            else:
                os.rename(f'traj{i}/in.{freq}_k', f"traj{0}/in.{freq}_k")
        os.chdir('../../..')

    def reformat_data(self, path):
        #print("Get current working directory:",os.getcwd())
        dat = open(f"{path}/dump.p2", "r")
        x, y, z, velx, vely, velz = [], [], [], [], [], [],
        lines = dat.readlines()

        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("ITEM: ATOMS"):
                i += 1
                if i < len(lines):
                    data_line = lines[i].strip()
                    parts = data_line.split()
                    try:
                        x.append(float(parts[2]))
                        y.append(float(parts[3]))
                        z.append(float(parts[4]))
                        velx.append(float(parts[5]))
                        vely.append(float(parts[6]))
                        velz.append(float(parts[7]))
                    except InDexError:
                        continue
            i += 1

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        velx = np.array(velx)
        vely = np.array(vely)
        velz = np.array(velz)

        header = f"Trajectory with external external force/potential. The timestep is {self.timestep}."
        np.savez(f"{path}/data", x=x, y=y, z=z, velx=velx, vely=vely, velz=velz, header=header)
       # np.savez(f"{path}/data", velx=velx, vely=vely, velz=velz, header=header)
 
    def calc_vacf(self, endname):
        dt = self.timestep
        for freq in self.Frequencies:
            corrs = []
            for i in range(self.N_simulations):
                print(i)
                # Construct the path to the data.npz file for each simulation
                filename = f"{self.dirname}/configurations/{freq}_k/traj{i}/data.npz"
                data = np.load(filename)
                velx = data["velz"]
                avg_velx = np.mean(velx)
                corr = correlate(velx - avg_velx, velx - avg_velx, mode = "full", method = "auto")
                corr = corr[len(corr)//2 :]
                corr /= corr[0]
                corrs.append(corr)
            corr = np.mean(corrs, axis=0)
            with open(f"{self.dirname}/analysis/{freq}_k/vacf/{endname}", "w") as f:
                f.write("# t\tcorr\n")
                for i in range(len(corr)):
                    f.write("{:f}\t{:.15e}\n".format(i*dt, corr[i]))
        


