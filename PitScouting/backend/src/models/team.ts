import { Model, DataTypes, Sequelize } from 'sequelize';

class Team extends Model {
  public id!: number;
  public teamNumber!: number;
  public autoScoreCoral!: boolean;
  public autoScoreAlgae!: boolean;
  public autoStartingPosition!: string;
  public teleopDealgifying!: boolean;
  public teleopPreference!: string;
  public scoringPreference!: string;
  public coralLevels!: string[];
  public endgameType!: string;
  public robotWidth!: number;
  public robotLength!: number;
  public robotHeight!: number;
  public robotWeight!: number;
  public drivetrainType!: string;
  public notes!: string;
  public imageUrl!: string;
  public mustStartSpecificPosition!: boolean;

  public static initialize(sequelize: Sequelize) {
    this.init({
      id: {
        type: DataTypes.INTEGER,
        autoIncrement: true,
        primaryKey: true,
      },
      teamNumber: {
        type: DataTypes.INTEGER,
        allowNull: false,
        unique: true,
      },
      autoScoreCoral: {
        type: DataTypes.BOOLEAN,
        defaultValue: false,
      },
      autoScoreAlgae: {
        type: DataTypes.BOOLEAN,
        defaultValue: false,
      },
      autoStartingPosition: {
        type: DataTypes.STRING,
        allowNull: true,
      },
      teleopDealgifying: {
        type: DataTypes.BOOLEAN,
        defaultValue: false,
      },
      teleopPreference: {
        type: DataTypes.STRING,
        allowNull: true,
      },
      scoringPreference: {
        type: DataTypes.STRING,
        allowNull: true,
      },
      coralLevels: {
        type: DataTypes.ARRAY(DataTypes.STRING),
        defaultValue: [],
        get() {
          const rawValue = this.getDataValue('coralLevels');
          return rawValue ? rawValue : [];
        },
        set(value: any) {
          if (typeof value === 'string') {
            try {
              this.setDataValue('coralLevels', JSON.parse(value));
            } catch {
              this.setDataValue('coralLevels', []);
            }
          } else if (Array.isArray(value)) {
            this.setDataValue('coralLevels', value);
          } else {
            this.setDataValue('coralLevels', []);
          }
        }
      },
      endgameType: {
        type: DataTypes.STRING,
        defaultValue: 'none',
      },
      robotWidth: {
        type: DataTypes.FLOAT,
        allowNull: true,
      },
      robotLength: {
        type: DataTypes.FLOAT,
        allowNull: true,
      },
      robotHeight: {
        type: DataTypes.FLOAT,
        allowNull: true,
      },
      robotWeight: {
        type: DataTypes.FLOAT,
        allowNull: true,
      },
      drivetrainType: {
        type: DataTypes.STRING,
        allowNull: true,
      },
      notes: {
        type: DataTypes.TEXT,
        allowNull: true,
      },
      imageUrl: {
        type: DataTypes.STRING,
        allowNull: true,
      },
      mustStartSpecificPosition: {
        type: DataTypes.BOOLEAN,
        defaultValue: false,
      }
    }, {
      sequelize,
      modelName: 'Team',
      tableName: 'teams'
    });
  }
}

export default Team; 