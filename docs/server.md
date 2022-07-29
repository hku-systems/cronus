# How to access our servers for artifact evaluation

## Steps

### Step 1: Connect to HKU CS gatekeeper

For security reasons, our department only allow access to the gatekeeper using SSH keys.

1. Create a key file (e.g., id_rsa) and copy the private key from our HotCRP response into it.

2. `chmod 400 [key_file_name]`.

3. `ssh -i [key_file_name] micro22@gatekeeper3.cs.hku.hk`.

### Step 2: Connect to our server

    ssh [server_account]@202.45.128.182

The server account and server password are listed in our HotCRP response.

## Descriptions about our cluster

Each machine in our cluster has two IP addresses, one for access from the
gatekeeper, one for interconnection among the machines, as shown in the below
table. Please use the `202.45.128.xxx` address to SSH machines from the
gatekeeper.

We have asked for access to one machine, but since the environment is not
completely clean, the experiment results may have some outliers if other people
happen to run tasks on the cluster at the same time. You may re-run the outlier
data points or simply ignore them when such contention happens.

## After accessing the cluster

We have helped to finish the setup for meeting prerequisite requirements and
creating the overlay network, so you can skip these steps in the instructions.

After cloning the codebase, you can skip all Deployment steps. We have updated
the readme file to mark the starting point.

Please be noted that different evaluators may not be able to run experiments at
the same time. This is because the docker containers started by other evaluators
will have the same IP address and container name.