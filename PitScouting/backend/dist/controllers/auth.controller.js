"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.login = exports.register = void 0;
const jsonwebtoken_1 = __importDefault(require("jsonwebtoken"));
const models_1 = require("../models");
const env_1 = __importDefault(require("../config/env"));
const register = async (req, res) => {
    try {
        const { name, email, password, teamNumber } = req.body;
        const existingUser = await models_1.User.findOne({ where: { email } });
        if (existingUser) {
            return res.status(400).json({ error: 'Email already exists' });
        }
        const user = await models_1.User.create({
            name,
            email,
            password,
            teamNumber,
        });
        const token = jsonwebtoken_1.default.sign({ id: user.id }, env_1.default.jwtSecret);
        return res.status(201).json({
            token,
            user: {
                id: user.id,
                name: user.name,
                email: user.email,
                teamNumber: user.teamNumber,
            },
        });
    }
    catch (error) {
        return res.status(400).json({ error: 'Error creating user' });
    }
};
exports.register = register;
const login = async (req, res) => {
    try {
        const { email, password } = req.body;
        const user = await models_1.User.findOne({ where: { email } });
        if (!user) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }
        const isValidPassword = await user.validatePassword(password);
        if (!isValidPassword) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }
        const token = jsonwebtoken_1.default.sign({ id: user.id }, env_1.default.jwtSecret);
        return res.json({
            token,
            user: {
                id: user.id,
                name: user.name,
                email: user.email,
                teamNumber: user.teamNumber,
            },
        });
    }
    catch (error) {
        return res.status(400).json({ error: 'Error logging in' });
    }
};
exports.login = login;
//# sourceMappingURL=auth.controller.js.map