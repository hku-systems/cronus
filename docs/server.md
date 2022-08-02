# How to access our servers for artifact evaluation

## Steps

### Step 1: Connect to HKU CS gatekeeper

For security reasons, our department only allow access to the gatekeeper using SSH keys.

*Updates: the department updates the security policy and allows only ssh forward now, please use instructions in step 3 and step 4 now.*

1. Create a key file (e.g., id_rsa) and copy the private key from our HotCRP response into it.

2. `chmod 400 [key_file_name]`.

3. create a ssh tunnel that forward localhost:2233 to the evaluation server (202.45.128.182): `ssh -L localhost:2233:202.45.128.182:22 micro22@gatekeeper3.cs.hku.hk -i idrsa_micro22`, please ***keep this terminal running*** to keep the tunnel alive.


### Step 2: Connect to our server

4. launch another terminal and login the evaluation server: `ssh -p 2233 jianyu@127.0.0.1`

The server account and server password are listed in our HotCRP response.

## Descriptions about our cluster

We have asked for access to one machine, but since the environment is not
completely clean, the experiment results may have some outliers if other people
happen to run tasks on the cluster at the same time. You may re-run the outlier
data points or simply ignore them when such contention happens.

## After accessing the cluster

Please be noted that different evaluators may not be able to run experiments at
the same time. This is because the docker containers started by other evaluators
will have the same IP address and container name.
