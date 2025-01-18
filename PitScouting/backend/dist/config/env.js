"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const dotenv_1 = __importDefault(require("dotenv"));
const path_1 = __importDefault(require("path"));
dotenv_1.default.config({ path: path_1.default.join(__dirname, '../../.env') });
const config = {
    port: process.env.PORT || 5001,
    jwtSecret: process.env.JWT_SECRET || 'your-secret-key',
    nodeEnv: process.env.NODE_ENV || 'development',
    uploadDir: path_1.default.join(__dirname, '../../uploads'),
    dbUrl: process.env.DATABASE_URL,
    corsOrigin: process.env.CORS_ORIGIN || 'http://localhost:3000',
    dbConfig: {
        development: {
            username: process.env.DB_USER || 'postgres',
            password: process.env.DB_PASSWORD || 'postgres',
            database: process.env.DB_NAME || 'scouting_db',
            host: process.env.DB_HOST || 'localhost',
            dialect: 'postgres'
        },
        production: {
            url: process.env.DATABASE_URL,
            dialect: 'postgres',
            dialectOptions: {
                ssl: {
                    require: true,
                    rejectUnauthorized: false
                }
            }
        }
    }
};
exports.default = config;
//# sourceMappingURL=env.js.map