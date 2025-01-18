import { Sequelize } from 'sequelize';

const env = process.env.NODE_ENV || 'development';
const databaseUrl = process.env.DATABASE_URL;

let sequelize: Sequelize;

if (env === 'production' && databaseUrl) {
  sequelize = new Sequelize(databaseUrl, {
    dialect: 'postgres',
    dialectOptions: {
      ssl: {
        require: true,
        rejectUnauthorized: false
      }
    }
  });
} else {
  sequelize = new Sequelize({
    database: process.env.DB_NAME || 'scouting_app',
    username: process.env.DB_USER || 'postgres',
    password: process.env.DB_PASSWORD || 'postgres',
    host: process.env.DB_HOST || 'localhost',
    dialect: 'postgres'
  });
}

export default sequelize; 