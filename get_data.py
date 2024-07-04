import paramiko
import os
import subprocess

def scp_directories(host, port, username, password, remote_dirs, local_dir, getcubes):
    # initialize the ssh client 
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # connect to inrainbows
        ssh.connect(host, port=port, username=username, password=password)

        # scp the desired directories 
        for key in remote_dirs.keys():
            # do not get scp the cube files 
            if key == 'cube' and getcubes == False:
                continue
            
            print(f"Obtaining {key} files.")
            # index the dict
            remote_path = remote_dirs[key]
            
            # define the local path
            local_path_rename = os.path.join(local_dir, f'{key}')

            # create the shell command
            scp_command = f"scp -r {username}@{host}:{remote_path} {local_path_rename}"

            # capture live output
            process = subprocess.Popen(scp_command, shell=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            while True:
                output = process.stdout.readline()
                if output == b'' and process.poll() is not None:
                    break
                if output:
                    print(output.strip().decode())
            rc = process.poll()
            
            print(f'Successfully copied {remote_dir} to {local_path}')

    except Exception as error:
        print(f"Error: {error}")




if __name__ == "__main__":
    # input the galaxy name and binning key
    galname = input("Enter Galaxy Name:")
    binkey = input("Enter binkey")

    # define remote directory of the data
    remote_dir = f"{galname}-{binkey}'

    # define the server and login information
    server = 'inrainbows'
    port = 22
    user = 'apitts'
    password = input(f"Enter password for {user}@{server}")

    # inrainbows paths to the mcmc and cube outputs
    mcmc_path = "NaImcmcIFU/muse/NaI_MCMC_output"
    cube_path = "mangadap_muse/outputs"

    # dict to hold remote paths
    remote_dirs = {
        'cube':os.path.join(cube_path,remote_dir),
        'mcmc':os.path.join(mcmc_path,remote_dir)
    }

    # define full path to local data
    local_path = os.path.abspath(__file__)
    local_dir = os.path.join(local_path,f"data/{galname}")

    # if directory for the galaxy does not exist, make it
    if not os.path.exists(local_path):
                os.mkdir(local_path)

    # getcubes flag to determine whether to pull the cubes files or just mcmc files
    getcubes = None
    while getcubes is None:
        ask = input("Include the Maps and Cubes? [Y/N]")

        if ask.lower() == 'y':
            getcubes = True
            print("MCMC and Cube files will be acquired.")

        elif ask.lower() == 'n':
            getcubes = False
            print("Only MCMC results will be acquired.")

        else:
            print("Invalid response.")
    
    # call the scp command
    scp_directories(server, port, user, password, remote_dirs, local_dir,
                    getcubes)