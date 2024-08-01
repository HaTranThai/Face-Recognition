#!/bin/bash

# Thêm đường dẫn cài đặt pip3 vào PATH
export PATH=$PATH:/home/bbsw/.local/bin
export PYTHONPATH=$PYTHONPATH:/home/bbsw/.local/lib/python3.6/site-packages

# Chạy lệnh sudo jetson_clocks
sudo jetson_clocks

# Bỏ qua lệnh nếu có lỗi
set -e

# Kiểm tra có hình ảnh dangling không
DANGLING_IMAGES=$(docker images -f "dangling=true" -q)

if [ -n "$DANGLING_IMAGES" ]; then
    # Nếu có hình ảnh dangling, xóa chúng
    echo "Dangling images found. Removing them..."
    docker rmi -f $DANGLING_IMAGES || true
fi

# Chạy docker-compose
sudo sync && sudo sysctl -w vm.drop_caches=3
docker-compose -f "/home/bbsw/Face_Recognition_JetNano/docker-compose.yml" up -d
# curl -X 'GET' 'http://cmdtech.ddns.net:2024/check_connection' -H 'accept: application/json'
# đường link sẽ trả về True nếu kết nối thành công, False nếu không kết nối được
# check_connection=$(curl -X 'GET' 'http://cmdtech.ddns.net:2024/check_connection' -H 'accept: application/json')
# if true; then
#     echo "Service đang chạy..." $check_connection
# else
#     echo "Service không chạy được..." $check_connection
#     docker-compose -f "/home/bbsw/Face_Recognition_JetNano/docker-compose.yml" down
#     sudo sync && sudo sysctl -w vm.drop_caches=3
#     docker-compose -f "/home/bbsw/Face_Recognition_JetNano/docker-compose.yml" up -d
# fi

# # Giữ script chạy
# while true; do
#     sleep 60
# done
