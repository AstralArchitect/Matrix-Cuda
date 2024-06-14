typedef struct {
    int x;
    int y;
} MatSize;

class Matrix
{
private:
    double *matrix_d;
    MatSize size;
public:
    Matrix(unsigned int dimX, unsigned int dimY);
    Matrix(const Matrix& other);
    ~Matrix();

    Matrix& operator=(const Matrix& other);
    Matrix operator*(Matrix const& matrix);
    Matrix operator+(Matrix const& matrix);
    Matrix operator+(double num);
    
    double get(int x, int y);
    void set(int x, int y, double value);

    double *getPtr();
};