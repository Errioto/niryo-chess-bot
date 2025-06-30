import json
import paramiko
from typing import Tuple, Optional
from pyniryo import PoseObject

def get_workspace_poses_ssh(
    hostname: str = "10.10.10.10",
    username: str = "niryo",
    password: Optional[str] = "robotics",
    key_filepath: Optional[str] = None,
    workspace_name: str = "ChessBoard"
) -> Tuple[PoseObject, PoseObject, PoseObject, PoseObject]:
    """
    Se connecte en SSH/SFTP au robot, lit
      ~/niryo_robot_saved_files/niryo_robot_workspaces/{workspace_name}.workspace
    et retourne un tuple de 4 PoseObject(x,y,z,rx,ry,rz).
    """
    # 1. SSH + SFTP
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs = {"hostname": hostname, "username": username}
    if key_filepath:
        connect_kwargs["key_filename"] = key_filepath
    else:
        connect_kwargs["password"] = password
    ssh.connect(**connect_kwargs)
    sftp = ssh.open_sftp()

    # 2. Trouver le home du user (ex: /home/niryo)
    home_dir = sftp.normalize('.')  # répertoire courant = home

    # 3. Construire le chemin absolu vers le fichier
    remote_path = (
        f"{home_dir}/niryo_robot_saved_files/"
        f"niryo_robot_workspaces/{workspace_name}.workspace"
    )

    # 4. Lecture JSON directement en mémoire
    try:
        with sftp.open(remote_path, 'r') as rf:
            data = json.load(rf)
    except IOError as e:
        raise FileNotFoundError(f"Impossible de lire {remote_path}: {e}")
    finally:
        sftp.close()
        ssh.close()

    # 5. Extraction et conversion en PoseObject
    raw_poses = data.get("robot_poses", [])
    if len(raw_poses) != 4:
        raise ValueError(f"On attend 4 poses, trouvé : {len(raw_poses)}")

    poses = []
    for position, orientation in raw_poses:
        poses.append(PoseObject(
            x  = position[0],
            y  = position[1],
            z  = position[2],
            roll = orientation[0],
            pitch = orientation[1],
            yaw = orientation[2],
        ))

    return tuple(poses)