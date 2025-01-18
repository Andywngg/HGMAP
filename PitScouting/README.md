# Team 1334 Pit Scouting App

A web application for scouting robots during FIRST Robotics Competition events.

## Features

- Team data collection with image upload
- Dashboard with filtering and sorting capabilities
- Export data to CSV
- Mobile-friendly interface

## Tech Stack

- Frontend: React with TypeScript, Tailwind CSS
- Backend: Node.js with Express, TypeScript
- Database: PostgreSQL
- Authentication: JWT

## Local Development Setup

1. Clone the repository:
```bash
git clone [your-repo-url]
cd pit-scouting
```

2. Install dependencies:
```bash
# Install backend dependencies
cd backend
npm install

# Install frontend dependencies
cd ../frontend
npm install
```

3. Set up environment variables:
Create `.env` files in both backend and frontend directories:

Backend `.env`:
```
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=scouting_app
DB_HOST=localhost
PORT=5001
JWT_SECRET=your_secret_key
```

4. Set up the database:
```bash
cd backend
npx sequelize-cli db:migrate
```

5. Start the development servers:
```bash
# Start both frontend and backend
npm run dev
```

## Deployment

### Database (Render)
1. Create a new PostgreSQL database on Render
2. Note the external database URL

### Backend (Render)
1. Create a new Web Service
2. Connect to your GitHub repository
3. Set build command: `npm install && npm run build`
4. Set start command: `npm start`
5. Add environment variables from `.env`

### Frontend (Vercel)
1. Import project from GitHub
2. Set environment variables
3. Deploy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Open a pull request

## License

MIT License 