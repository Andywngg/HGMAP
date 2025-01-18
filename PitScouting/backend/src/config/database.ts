import { Sequelize } from 'sequelize';
import env from './env';

const sequelize = new Sequelize(env.dbName, env.dbUser, env.dbPassword, {
  host: env.dbHost,
  dialect: 'postgres',
  logging: false,
});

export default sequelize; 