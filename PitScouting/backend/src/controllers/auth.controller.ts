import { Request, Response } from 'express';
import jwt from 'jsonwebtoken';
import { User } from '../models';
import env from '../config/env';

export const register = async (req: Request, res: Response): Promise<Response> => {
  try {
    const { name, email, password, teamNumber } = req.body;

    const existingUser = await User.findOne({ where: { email } });
    if (existingUser) {
      return res.status(400).json({ error: 'Email already exists' });
    }

    const user = await User.create({
      name,
      email,
      password,
      teamNumber,
    });

    const token = jwt.sign({ id: user.id }, env.jwtSecret);

    return res.status(201).json({
      token,
      user: {
        id: user.id,
        name: user.name,
        email: user.email,
        teamNumber: user.teamNumber,
      },
    });
  } catch (error) {
    console.error('Registration error:', error);
    return res.status(400).json({ error: 'Error creating user' });
  }
};

export const login = async (req: Request, res: Response): Promise<Response> => {
  try {
    const { email, password } = req.body;
    console.log('Login attempt for email:', email);

    // Hardcoded admin credentials
    if (email === '1334admin@gmail.com' && password === 'otisit!!!') {
      const token = jwt.sign({ id: 1, isAdmin: true }, env.jwtSecret);
      return res.json({
        token,
        user: {
          id: 1,
          name: 'Team 1334 Admin',
          email: '1334admin@gmail.com',
          teamNumber: 1334,
        },
      });
    }

    const user = await User.findOne({ where: { email } });
    if (!user) {
      console.log('User not found for email:', email);
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const isValidPassword = await user.validatePassword(password);
    console.log('Password validation result:', isValidPassword);
    
    if (!isValidPassword) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const token = jwt.sign({ id: user.id }, env.jwtSecret);

    return res.json({
      token,
      user: {
        id: user.id,
        name: user.name,
        email: user.email,
        teamNumber: user.teamNumber,
      },
    });
  } catch (error) {
    console.error('Login error:', error);
    return res.status(400).json({ error: 'Error logging in' });
  }
}; 