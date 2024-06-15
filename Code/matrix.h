typedef struct {
    int x;
    int y;
} MatSize;

class Matrix
{
private:
    float *matrix_d;
    MatSize size;

    Matrix(unsigned int dimX, unsigned int dimY);
public:
    Matrix(unsigned int dimX, unsigned int dimY, float startValue);
    Matrix(const Matrix& other);
    ~Matrix();

    Matrix& operator=(const Matrix& other);
    
    Matrix operator*(Matrix const& matrix);
    Matrix operator*=(Matrix const& matrix);

    Matrix operator+(Matrix const& matrix);
    Matrix operator+=(Matrix const& matrix);

    Matrix operator+(float num);
    Matrix operator+=(float num);
    
    float get(int x, int y);
    void set(int x, int y, float value);

    float *getPtr();
};