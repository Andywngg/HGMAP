import { Sequelize } from 'sequelize';
import env from './env';

const { dbConfig, nodeEnv } = env;

type DevConfig = {
  username: string;
  password: string;
  database: string;
  host: string;
  dialect: string;
};

type ProdConfig = {
  url: string | undefined;
  dialect: string;
  dialectOptions: {
    ssl: {
      require: boolean;
      rejectUnauthorized: boolean;
    };
  };
};

const config = dbConfig[nodeEnv as keyof typeof dbConfig] as (DevConfig | ProdConfig);

let sequelize: Sequelize;

if (nodeEnv === 'production' && (config as ProdConfig).url) {
  sequelize = new Sequelize((config as ProdConfig).url!, {
    dialect: 'postgres',
    dialectOptions: {
      ssl: {
        require: true,
        rejectUnauthorized: false
      }
    },
    logging: false
  });
} else {
  const devConfig = config as DevConfig;
  sequelize = new Sequelize(
    devConfig.database,
    devConfig.username,
    devConfig.password,
    {
      host: devConfig.host,
      dialect: 'postgres',
      logging: false
    }
  );
}

export default sequelize; 