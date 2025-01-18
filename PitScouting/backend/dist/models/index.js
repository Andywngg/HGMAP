"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.Team = exports.User = exports.sequelize = void 0;
const sequelize_1 = require("sequelize");
const config_1 = __importDefault(require("../config/config"));
const user_1 = require("./user");
Object.defineProperty(exports, "User", { enumerable: true, get: function () { return user_1.User; } });
const team_1 = __importDefault(require("./team"));
exports.Team = team_1.default;
const env = process.env.NODE_ENV || 'development';
const dbConfig = config_1.default[env];
let sequelize;
if ('use_env_variable' in dbConfig && dbConfig.use_env_variable) {
    const dbUrl = process.env[dbConfig.use_env_variable];
    if (!dbUrl)
        throw new Error(`Environment variable ${dbConfig.use_env_variable} is not set`);
    exports.sequelize = sequelize = new sequelize_1.Sequelize(dbUrl, Object.assign(Object.assign({}, dbConfig), { dialect: 'postgres' }));
}
else {
    const config = dbConfig;
    exports.sequelize = sequelize = new sequelize_1.Sequelize(config.database, config.username, config.password, Object.assign(Object.assign({}, dbConfig), { dialect: 'postgres' }));
}
user_1.User.initialize(sequelize);
team_1.default.initialize(sequelize);
//# sourceMappingURL=index.js.map