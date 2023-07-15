# From workspace rootdir: sh ./scripts/build_docker.sh
docker build -t omneer_server:v1 --platform linux/amd64 -f Dockerfile .
# Kill any process using port 8080
sudo fuser -k 8080/tcp
# Run 
docker run -it --rm -v $(pwd)/server:/app/server -p 8080:8080 --name omneer_server omneer_server:v1