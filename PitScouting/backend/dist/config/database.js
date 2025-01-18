"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const sequelize_1 = require("sequelize");
const env_1 = __importDefault(require("./env"));
const { dbConfig, nodeEnv } = env_1.default;
const config = dbConfig[nodeEnv];
let sequelize;
if (nodeEnv === 'production' && config.url) {
    sequelize = new sequelize_1.Sequelize(config.url, {
        dialect: 'postgres',
        dialectOptions: {
            ssl: {
                require: true,
                rejectUnauthorized: false
            }
        },
        logging: false
    });
}
else {
    const devConfig = config;
    sequelize = new sequelize_1.Sequelize(devConfig.database, devConfig.username, devConfig.password, {
        host: devConfig.host,
        dialect: 'postgres',
        logging: false
    });
}
exports.default = sequelize;
//# sourceMappingURL=database.js.map