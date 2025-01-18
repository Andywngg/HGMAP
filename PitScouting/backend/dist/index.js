"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const cors_1 = __importDefault(require("cors"));
const path_1 = __importDefault(require("path"));
const models_1 = require("./models");
const env_1 = __importDefault(require("./config/env"));
const auth_1 = __importDefault(require("./routes/auth"));
const teams_1 = __importDefault(require("./routes/teams"));
const app = (0, express_1.default)();
app.use((0, cors_1.default)());
app.use(express_1.default.json());
app.use('/uploads', express_1.default.static(path_1.default.join(__dirname, '../uploads')));
app.use('/api/auth', auth_1.default);
app.use('/api/teams', teams_1.default);
app.use((err, _req, res, _next) => {
    console.error(err.stack);
    res.status(500).json({ error: 'Something broke!' });
});
async function startServer() {
    try {
        await models_1.sequelize.authenticate();
        console.log('Database connected successfully');
        app.listen(env_1.default.port, () => {
            console.log(`Server running on port ${env_1.default.port}`);
        });
    }
    catch (error) {
        console.error('Unable to connect to the database:', error);
    }
}
startServer();
exports.default = app;
//# sourceMappingURL=index.js.map