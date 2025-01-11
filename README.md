# Run the app directly
1. Open a terminal frontend dir and run: npm install; npm run dev
2. Open another terminal in backend dir and run: uvicorn app.app:app --reload  

# Run the app via Docker
1. Open a terminal in the root folder (TwoTowerInterface) and run: docker-compose up --build
2. After you've added the TwoTower Model to ./backend folder and linked it to fastapi run: docker push alberto1alberto/mlx:two_tower_model_with_interface
3. To check the running app see url: http://135.181.83.116:8080/
4. ssh into hetzner server:  ssh -i C:\Users\Alber\.ssh\hetzner root@135.181.83.116
5. uvicorn backend.app.app:app --reload --port 8000
6. docker exec -it 7c8c7ec9ea02 psql -U postgres -d postgres
