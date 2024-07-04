import subprocess
import paramiko
from scp import SCPClient
import os



def create_ssh_client(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

def scp_get_directories(ssh_client, remote_dirs, local_path):
    with SCPClient(sshclient.get_transport()) as scp:
        for remote_dir in remote_dirs:
            scp.get(remote_dir, local_path=local_path, recursive=True)



if __name__ == "__main__":
    print("Warning: Assuming desired data is SQUARE0.6")
    galname = input("Enter Galaxy Name:")


    server = 'inrainbows'
    port = 22
    user = 'apitts'
    password = input(f"Enter your password for {server}")

    remote_dirs = []

    dataflag = None
    while dataflag is None:
        dataflag = input("Which data would you like? [CUBES/MCMC/ALL]")

        if dataflag.lower() == 'cubes':
            remote_dirs.append(f"mangadap_muse/outputs/{galname}-SQUARE0.6")

        elif dataflag.lower() == 'mcmc':
            remote_dirs.append(f"NaImcmcIFU/muse/NaI_MCMC_output/{galname}-SQUARE0.6")

        elif dataflag.lower() == 'all':
            remote_dirs.append(f"NaImcmcIFU/muse/NaI_MCMC_output/{galname}-SQUARE0.6")
            remote_dirs.append(f"mangadap_muse/outputs/{galname}-SQUARE0.6")

        else:
            print('Invalid response. Please respond with the following options: CUBES, MCMC, ALL')
            dataflag = None


    local_gal_path = f"~/Work/data/{galname}"
    if not os.path.exists(local_path):
        os.mkdir(local_path)

    local_cube_path = os.path.join(local_gal_path, "cubes")
    if not os.path.exists(local_cube_path):
        os.mkdir(local_cube_path)

    local_mcmc_path = os.path.join(local_gal_path, "mcmc")
    if not os.path.exists(local_mcmc_path):
        os.mkdir(local_mcmc_path)

    directories = {
        
    }
    