import os
from modelscope.hub.api import HubApi

# 1. 登录
api = HubApi()
# 使用OAuth2认证，从URL中提取认证信息
api.login('ms-bb1493af-1415-41e0-be12-937342d5e15a')

# 2. 上传目录下的所有文件
api.upload_folder(
    repo_id='xsheng9867/mydata',
    folder_path='/home/haris/raid0/shared/haris/mydata_20251231',
    commit_message='update_by_haris: version 0.0: initialize',
    repo_type='dataset',
    max_workers=os.cpu_count() - 2,
)
print(f'Uploaded dataset folder successfully!')
